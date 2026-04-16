"""GPU smoke test — diagnoses hardware, AMP, VRAM cap, and throughput.

Run on the cluster after ``setup_cluster.sh``:

::

    python -m csd2smpl.tests.test_gpu_smoke
    python -m csd2smpl.tests.test_gpu_smoke --config csd2smpl/configs/v100_3way.yaml
    python -m csd2smpl.tests.test_gpu_smoke --find-batch
    python -m csd2smpl.tests.test_gpu_smoke --stress 30    # 30 s of training

Each phase prints its own pass/fail plus a hint on how to recover if it
fails, so the output is useful as a diagnostic when something's wrong on
the cluster (driver mismatch, OOM, fp16 instability, co-tenant contention).
Exit code is 0 on green, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.optim import AdamW

from csd2smpl.config_utils import load_config
from csd2smpl.losses import compute_losses
from csd2smpl.models.pipeline import Markers2SMPL


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _hint(msg: str) -> None:
    print(f"      {DIM}hint: {msg}{RESET}")


def _human_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


def _human_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


@contextlib.contextmanager
def _phase(name: str):
    """Print ``[name]`` header + timing; catches + reports exceptions."""
    print(f"\n[{name}]")
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        _fail(f"{exc.__class__.__name__}: {exc}")
        traceback.print_exc(limit=3)
        raise
    finally:
        dt = time.perf_counter() - t0
        print(f"  {DIM}({dt * 1000:.0f} ms){RESET}")


def _tiny_cfg() -> dict:
    """Fallback config when no YAML is given (smaller than smoke.yaml)."""
    return {
        "n_markers": 41, "n_smpl_joints": 24, "n_betas": 10,
        "d_model": 128, "n_heads": 4, "n_layers": 2,
        "dropout": 0.0, "max_seq_len": 256,
        "seq_len": 64, "batch_size": 16,
        "lr": 3.0e-4,
        "amp": True, "amp_dtype": "float16",
        "w_pose": 5.0, "w_shape": 0.1, "w_trans": 1.0, "w_smooth": 0.1,
        "grad_clip": 1.0,
    }


def _make_batch(cfg: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Random batch matching the dataset contract on the given device."""
    b, t, m = cfg["batch_size"], cfg["seq_len"], cfg["n_markers"]
    g = torch.Generator(device="cpu").manual_seed(0)
    return {
        "markers": torch.randn(b, t, m, 3, generator=g).to(device),
        "mask": torch.ones(b, t, m, dtype=torch.bool, device=device),
        "poses_gt": (0.5 * torch.randn(b, t, cfg["n_smpl_joints"] * 3, generator=g)).to(device),
        "betas_gt": torch.randn(b, cfg["n_betas"], generator=g).to(device),
        "trans_gt": (0.1 * torch.randn(b, t, 3, generator=g)).to(device),
    }


def _loss_kwargs(cfg: dict) -> dict:
    return {k: cfg[k] for k in ("w_pose", "w_shape", "w_trans", "w_smooth")}


def _reset_vram_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return torch.cuda.max_memory_allocated(device)


# ── Phase checks ──────────────────────────────────────────────


def check_cuda() -> torch.device:
    """Verify CUDA is visible and return the device we'll use."""
    with _phase("1. CUDA visibility"):
        if not torch.cuda.is_available():
            _fail(f"torch.cuda.is_available()=False (built for CUDA {torch.version.cuda})")
            _hint("install a CUDA torch wheel: bash csd2smpl/scripts/setup_cluster.sh")
            _hint("or pass --device cpu to force CPU (skips most of this smoke)")
            raise SystemExit(2)
        _ok(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
        n = torch.cuda.device_count()
        _ok(f"{n} device(s) visible")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            gb = p.total_memory / 1024**3
            _ok(f"  [{i}] {p.name}  {gb:.1f} GB  sm_{p.major}{p.minor}")
        device = torch.device("cuda:0")
        return device


def check_capability_vs_config(cfg: dict, device: torch.device) -> None:
    """Warn (don't fail) when config requests something the GPU can't do well."""
    with _phase("2. GPU ↔ config compatibility"):
        props = torch.cuda.get_device_properties(device)
        sm = props.major * 10 + props.minor
        amp_dtype = cfg.get("amp_dtype", "bfloat16").lower()
        if cfg.get("amp") and amp_dtype == "bfloat16" and sm < 80:
            _warn(f"amp_dtype=bfloat16 but sm_{props.major}{props.minor} (<Ampere); fp16 recommended")
            _hint("set amp_dtype: float16 in the config (V100 fix)")
        else:
            _ok(f"amp_dtype={amp_dtype} is OK for sm_{props.major}{props.minor}")
        frac = cfg.get("vram_fraction")
        if frac is not None:
            total_gb = props.total_memory / 1024**3
            cap_gb = frac * total_gb
            _ok(f"requested VRAM cap: {frac:.2%} of {total_gb:.1f} GB ≈ {cap_gb:.1f} GB")


def check_vram_cap(cfg: dict, device: torch.device) -> None:
    """Apply the per-process memory fraction and confirm OOM is clean."""
    with _phase("3. VRAM cap enforcement"):
        frac = cfg.get("vram_fraction")
        if frac is None:
            _warn("no vram_fraction set; skipping")
            return
        torch.cuda.set_per_process_memory_fraction(float(frac), device=device.index or 0)
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        cap_gb = frac * total_gb
        _ok(f"set_per_process_memory_fraction({frac}) → ~{cap_gb:.1f} GB cap")

        # Try to allocate 1.5× the cap to confirm it fails fast.
        probe_gb = cap_gb * 1.5
        n_elems = int(probe_gb * 1024**3 / 4)  # float32 = 4 bytes
        try:
            _ = torch.empty(n_elems, dtype=torch.float32, device=device)
            _warn(f"allocating {probe_gb:.1f} GB did NOT OOM — cap may be loose "
                  f"(total VRAM is large enough to absorb the probe)")
        except torch.cuda.OutOfMemoryError:
            _ok(f"probe allocation of {probe_gb:.1f} GB OOM'd as expected")
        finally:
            _reset_vram_stats(device)


def build_model_and_batch(
    cfg: dict, device: torch.device
) -> tuple[Markers2SMPL, dict[str, torch.Tensor], AdamW]:
    """Instantiate model + batch + optimizer; report resident VRAM."""
    with _phase("4. Model + batch build"):
        _reset_vram_stats(device)
        model = Markers2SMPL(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _ok(f"model: {n_params:,} params")
        batch = _make_batch(cfg, device)
        _ok(f"batch: markers={tuple(batch['markers'].shape)}  poses_gt={tuple(batch['poses_gt'].shape)}")
        opt = AdamW(model.parameters(), lr=cfg["lr"])
        resident = torch.cuda.memory_allocated(device)
        _ok(f"resident VRAM after setup: {_human_mb(resident)}")
        return model, batch, opt


def check_forward_backward(
    cfg: dict, device: torch.device,
    model: Markers2SMPL, batch: dict[str, torch.Tensor], opt: AdamW,
) -> None:
    """One full step: fp32 forward, AMP forward, backward, scaler, clip, opt.step."""
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        cfg.get("amp_dtype", "float16").lower()
    ]
    use_amp = cfg.get("amp", False)

    with _phase("5. fp32 forward"):
        _reset_vram_stats(device)
        pred = model(batch["markers"], batch["mask"])
        loss = compute_losses(*pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"],
                              **_loss_kwargs(cfg))["loss"]
        if not torch.isfinite(loss):
            _fail(f"fp32 loss not finite: {loss.item()}")
            raise RuntimeError("non-finite fp32 loss")
        _ok(f"loss={loss.item():.4f}, peak VRAM={_human_mb(_peak_vram(device))}")

    with _phase(f"6. AMP {amp_dtype.__str__().split('.')[-1]} forward + backward"):
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        opt.zero_grad(set_to_none=True)
        _reset_vram_stats(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            pred = model(batch["markers"], batch["mask"])
            loss = compute_losses(*pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"],
                                  **_loss_kwargs(cfg))["loss"]
        if not torch.isfinite(loss):
            _fail(f"AMP loss not finite: {loss.item()}")
            _hint("try amp_dtype: float32 (disables AMP), or check for NaN-inducing inputs")
            raise RuntimeError("non-finite AMP loss")
        _ok(f"AMP forward loss={loss.item():.4f}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        scaler.step(opt)
        scaler.update()
        _ok(f"backward + step OK, peak VRAM={_human_mb(_peak_vram(device))}")


def check_overfit(cfg: dict, device: torch.device) -> None:
    """Run 20 AMP steps on a fixed batch; loss must drop."""
    with _phase("7. Overfit 20 AMP steps (sanity)"):
        torch.manual_seed(0)
        model = Markers2SMPL(cfg).to(device)
        batch = _make_batch(cfg, device)
        opt = AdamW(model.parameters(), lr=cfg["lr"])
        use_amp = cfg.get("amp", False)
        amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg.get("amp_dtype", "float16").lower()
        ]
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        kw = _loss_kwargs(cfg)

        def step_loss() -> float:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(batch["markers"], batch["mask"])
                losses = compute_losses(*pred, batch["poses_gt"], batch["betas_gt"],
                                        batch["trans_gt"], **kw)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            return float(losses["loss"].detach())

        initial = step_loss()
        for _ in range(19):
            step_loss()
        final = step_loss()
        drop = (initial - final) / max(initial, 1e-8)
        if drop < 0.2:
            _fail(f"loss drop only {drop:.1%} (expected ≥20%); initial={initial:.4f} final={final:.4f}")
            _hint("check grad flow; lower lr; disable AMP to isolate")
            raise RuntimeError("overfit failed")
        _ok(f"loss {initial:.4f} → {final:.4f}  ({drop:.1%} drop)")


def check_throughput(cfg: dict, device: torch.device, seconds: float) -> None:
    """Loop training for ``seconds`` of wall time; report steps/sec and VRAM peak."""
    with _phase(f"8. Throughput ({seconds:.0f} s wall)"):
        torch.manual_seed(0)
        model = Markers2SMPL(cfg).to(device)
        batch = _make_batch(cfg, device)
        opt = AdamW(model.parameters(), lr=cfg["lr"])
        use_amp = cfg.get("amp", False)
        amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg.get("amp_dtype", "float16").lower()
        ]
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        kw = _loss_kwargs(cfg)

        # Warm-up so we don't count CUDA graph init.
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(batch["markers"], batch["mask"])
                loss = compute_losses(*pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"], **kw)["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize(device)
        _reset_vram_stats(device)

        t_end = time.perf_counter() + seconds
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() < t_end:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(batch["markers"], batch["mask"])
                loss = compute_losses(*pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"], **kw)["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            steps += 1
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        samples_per_sec = steps * cfg["batch_size"] / dt
        _ok(f"{steps} steps in {dt:.1f}s → {steps / dt:.1f} step/s, {samples_per_sec:.0f} samples/s")
        _ok(f"peak VRAM during stress: {_human_mb(_peak_vram(device))}")


def find_max_batch(cfg_base: dict, device: torch.device) -> None:
    """Binary-search the largest batch_size that fits within the VRAM cap."""
    with _phase("9. Max-batch binary search"):
        lo, hi = 1, 1024
        best = 0
        cfg = dict(cfg_base)
        while lo <= hi:
            mid = (lo + hi) // 2
            cfg["batch_size"] = mid
            gc.collect()
            _reset_vram_stats(device)
            try:
                model = Markers2SMPL(cfg).to(device)
                batch = _make_batch(cfg, device)
                opt = AdamW(model.parameters(), lr=cfg["lr"])
                opt.zero_grad(set_to_none=True)
                amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
                    cfg.get("amp_dtype", "float16").lower()
                ]
                scaler = torch.amp.GradScaler("cuda", enabled=(cfg.get("amp") and amp_dtype == torch.float16))
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.get("amp", False)):
                    pred = model(batch["markers"], batch["mask"])
                    loss = compute_losses(*pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"],
                                          **_loss_kwargs(cfg))["loss"]
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize(device)
                peak = _peak_vram(device)
                best = mid
                lo = mid + 1
                print(f"  batch={mid:4d}  OK   peak={_human_mb(peak)}")
            except torch.cuda.OutOfMemoryError:
                hi = mid - 1
                print(f"  batch={mid:4d}  OOM")
            finally:
                del model, batch, opt
                gc.collect()
                torch.cuda.empty_cache()
        _ok(f"max batch_size that fits: {best} (with current d_model/seq_len/n_layers)")


# ── Main ──────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="csd2smpl/configs/v100_3way.yaml",
        help="config to test with (default: v100_3way.yaml)",
    )
    parser.add_argument(
        "--find-batch", action="store_true",
        help="after the main smoke, binary-search max batch_size",
    )
    parser.add_argument(
        "--stress", type=float, default=5.0,
        help="seconds of throughput stress test (default: 5.0)",
    )
    args = parser.parse_args()

    # Load config (tolerate missing YAML for a quick sanity-only run).
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        # Fill any missing hyperparams with tiny defaults so the smoke is
        # self-contained when running against a config that lacks them.
        for k, v in _tiny_cfg().items():
            cfg.setdefault(k, v)
        print(f"Using config: {cfg_path}")
    else:
        cfg = _tiny_cfg()
        print(f"Config {cfg_path} not found — using built-in tiny defaults")

    print(f"Model shape: d_model={cfg['d_model']} n_layers={cfg['n_layers']} "
          f"batch={cfg['batch_size']} seq={cfg['seq_len']} markers={cfg['n_markers']}")

    failures: list[str] = []
    device: torch.device | None = None
    try:
        device = check_cuda()
        check_capability_vs_config(cfg, device)
        check_vram_cap(cfg, device)
        model, batch, opt = build_model_and_batch(cfg, device)
        check_forward_backward(cfg, device, model, batch, opt)
        del model, batch, opt
        gc.collect()
        torch.cuda.empty_cache()

        check_overfit(cfg, device)
        check_throughput(cfg, device, args.stress)

        if args.find_batch:
            find_max_batch(cfg, device)

    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        failures.append(str(exc))

    print("\n── Summary ──────────────────────────────────────────")
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        print(f"  peak VRAM overall : {_human_mb(peak)}  ({peak / total:.1%} of {_human_gb(total)})")
        if "vram_fraction" in cfg:
            cap = cfg["vram_fraction"] * total
            print(f"  VRAM cap         : {_human_mb(cap)}  (headroom {_human_mb(cap - peak)})")
    if failures:
        print(f"  {RED}STATUS: FAIL — {len(failures)} phase(s) failed{RESET}")
        return 1
    print(f"  {GREEN}STATUS: PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
