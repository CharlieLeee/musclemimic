"""Training loop for the markers→SMPL pipeline.

Usage
-----
::

    # CPU smoke
    python -m csd2smpl.train --config csd2smpl/configs/default.yaml

    # GPU run
    python -m csd2smpl.train --config csd2smpl/configs/gpu.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from csd2smpl.config_utils import load_config
from csd2smpl.data.c3d_dataset import MarkerDataset
from csd2smpl.losses import compute_losses
from csd2smpl.models.pipeline import Markers2SMPL


def _resolve_device(cfg: dict) -> torch.device:
    """Use ``cfg["device"]`` if set, else cuda if available, else cpu."""
    if "device" in cfg:
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_amp(cfg: dict, device: torch.device) -> tuple[bool, torch.dtype]:
    """Decide if autocast is on and which dtype to use (bf16 default on GPU)."""
    if not cfg.get("amp", False) or device.type != "cuda":
        return False, torch.float32
    name = cfg.get("amp_dtype", "bfloat16").lower()
    return True, {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _enforce_vram_cap(cfg: dict, device: torch.device) -> None:
    """Apply ``cfg['vram_fraction']`` as a hard per-process VRAM cap.

    This is the mechanism for sharing a V100 across N tenants: each process
    sets ``fraction = per_process_GB / total_GB``. PyTorch refuses
    allocations that would breach the cap (clean OOM rather than evicting
    a co-tenant).
    """
    frac = cfg.get("vram_fraction")
    if frac is None or device.type != "cuda":
        return
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"vram_fraction must be in (0, 1]; got {frac}")
    torch.cuda.set_per_process_memory_fraction(float(frac), device=device.index or 0)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"VRAM cap: {frac:.2%} of {total_gb:.1f} GB ≈ {frac * total_gb:.1f} GB")


def _run_loss(
    model: Markers2SMPL, batch: dict[str, torch.Tensor], loss_kwargs: dict
) -> dict[str, torch.Tensor]:
    pred = model(batch["markers"], batch["mask"])
    return compute_losses(*pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"], **loss_kwargs)


def train(cfg: dict) -> None:
    """Run the full train / validate loop described in ``cfg``."""
    device = _resolve_device(cfg)
    use_amp, amp_dtype = _resolve_amp(cfg, device)
    _enforce_vram_cap(cfg, device)

    cache_in_ram = cfg.get("cache_in_ram", True)
    train_set = MarkerDataset(
        cfg["data_root"], split="train",
        seq_len=cfg["seq_len"], stride=cfg["stride"],
        cache_in_ram=cache_in_ram,
    )
    val_set = MarkerDataset(
        cfg["data_root"], split="val",
        seq_len=cfg["seq_len"], stride=cfg["seq_len"],
        cache_in_ram=cache_in_ram,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg.get("num_workers", 4),
        pin_memory=pin, persistent_workers=cfg.get("num_workers", 4) > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg.get("num_workers_val", 2),
        pin_memory=pin, persistent_workers=cfg.get("num_workers_val", 2) > 0,
    )

    model = Markers2SMPL(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {n_params:,} trainable params on {device}; "
        f"AMP={use_amp} ({amp_dtype.__str__().split('.')[-1] if use_amp else 'fp32'})"
    )

    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])
    # GradScaler is only needed for fp16 (bf16 doesn't underflow).
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    grad_clip = cfg.get("grad_clip", 1.0)
    log_every = cfg.get("log_every", 50)
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_kwargs = {
        "w_pose": cfg["w_pose"], "w_shape": cfg["w_shape"],
        "w_trans": cfg["w_trans"], "w_smooth": cfg["w_smooth"],
    }

    best_val = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                losses = _run_loss(model, batch, loss_kwargs)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            train_loss += losses["loss"].item()
            if step % log_every == 0:
                print(
                    f"  epoch {epoch+1:03d} step {step:05d}  "
                    f"total={losses['loss'].item():.4f}  "
                    f"pose={losses['L_pose'].item():.4f} "
                    f"shape={losses['L_shape'].item():.4f} "
                    f"trans={losses['L_trans'].item():.4f}"
                )

        sched.step()
        train_loss /= max(len(train_loader), 1)
        train_t = time.perf_counter() - t0

        if (epoch + 1) % cfg.get("val_every", 1) == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        losses = _run_loss(model, batch, loss_kwargs)
                    val_loss += losses["loss"].item()
            val_loss /= max(len(val_loader), 1)
        else:
            val_loss = float("nan")

        print(f"Epoch {epoch+1:03d}  train={train_loss:.4f}  val={val_loss:.4f}  t={train_t:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  saved best checkpoint (val={val_loss:.4f})")

    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    print(f"Done. Best val={best_val:.4f}, last checkpoint at {ckpt_dir/'last.pt'}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="csd2smpl/configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
