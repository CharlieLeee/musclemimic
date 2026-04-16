"""Smoke test for the markers→SMPL pipeline.

Goals
-----
1. Build a tiny pipeline (small d_model, 2 layers).
2. Generate synthetic random markers + matching SMPL targets — no AMASS,
   no smplx, no SMPL .pkl.
3. Verify shapes, finite gradients, and that loss decreases when overfitting
   one batch for ~50 steps. Proves architecture wires together correctly and
   gradients flow through the 6D→rotmat→axis-angle path.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW

from csd2smpl.losses import compute_losses
from csd2smpl.models.pipeline import Markers2SMPL


SMOKE_CFG = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    n_markers: int,
    n_smpl_joints: int,
    n_betas: int,
    device: torch.device,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Random markers + SMPL targets matching the dataset contract."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "markers": torch.randn(batch_size, seq_len, n_markers, 3, generator=g).to(device),
        "mask": torch.ones(batch_size, seq_len, n_markers, dtype=torch.bool, device=device),
        "poses_gt": (0.5 * torch.randn(batch_size, seq_len, n_smpl_joints * 3, generator=g)).to(device),
        "betas_gt": torch.randn(batch_size, n_betas, generator=g).to(device),
        "trans_gt": (0.1 * torch.randn(batch_size, seq_len, 3, generator=g)).to(device),
    }


def _load_cfg() -> dict:
    with SMOKE_CFG.open() as f:
        return yaml.safe_load(f)


def _loss_kwargs(cfg: dict) -> dict:
    return {
        "w_pose": cfg["w_pose"], "w_shape": cfg["w_shape"],
        "w_trans": cfg["w_trans"], "w_smooth": cfg["w_smooth"],
    }


def test_forward_shapes_and_finite_gradients() -> None:
    """One forward + backward pass produces expected shapes and finite grads."""
    cfg = _load_cfg()
    device = torch.device("cpu")
    model = Markers2SMPL(cfg).to(device)
    batch = make_synthetic_batch(
        cfg["batch_size"], cfg["seq_len"],
        cfg["n_markers"], cfg["n_smpl_joints"], cfg["n_betas"], device,
    )

    pred_poses, pred_betas, pred_trans = model(batch["markers"], batch["mask"])

    expected_poses = (cfg["batch_size"], cfg["seq_len"], cfg["n_smpl_joints"] * 3)
    expected_betas = (cfg["batch_size"], cfg["n_betas"])
    expected_trans = (cfg["batch_size"], cfg["seq_len"], 3)
    assert tuple(pred_poses.shape) == expected_poses, (
        f"pred_poses shape: expected {expected_poses}, got {tuple(pred_poses.shape)}"
    )
    assert tuple(pred_betas.shape) == expected_betas, (
        f"pred_betas shape: expected {expected_betas}, got {tuple(pred_betas.shape)}"
    )
    assert tuple(pred_trans.shape) == expected_trans, (
        f"pred_trans shape: expected {expected_trans}, got {tuple(pred_trans.shape)}"
    )

    losses = compute_losses(
        pred_poses, pred_betas, pred_trans,
        batch["poses_gt"], batch["betas_gt"], batch["trans_gt"],
        **_loss_kwargs(cfg),
    )
    assert torch.isfinite(losses["loss"]), f"loss not finite: {losses['loss'].item()}"

    losses["loss"].backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert grad_norms, "no gradients computed"
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), (
        f"non-finite gradient detected; max={max(grad_norms)}, min={min(grad_norms)}"
    )


def test_overfit_single_batch_reduces_loss() -> None:
    """Loss on a fixed batch must drop by >=50% after 50 optimizer steps."""
    cfg = _load_cfg()
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = Markers2SMPL(cfg).to(device)
    batch = make_synthetic_batch(
        cfg["batch_size"], cfg["seq_len"],
        cfg["n_markers"], cfg["n_smpl_joints"], cfg["n_betas"], device,
    )

    opt = AdamW(model.parameters(), lr=cfg["lr"])
    kw = _loss_kwargs(cfg)

    def step_loss() -> dict[str, torch.Tensor]:
        return compute_losses(
            *model(batch["markers"], batch["mask"]),
            batch["poses_gt"], batch["betas_gt"], batch["trans_gt"], **kw,
        )

    model.train()
    initial = step_loss()["loss"].item()
    for _ in range(50):
        opt.zero_grad()
        loss = step_loss()["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final = step_loss()["loss"].item()
    drop = (initial - final) / initial
    assert drop >= 0.5, (
        f"expected >=50% loss drop after 50 steps; got {drop:.1%} "
        f"(initial={initial:.4f}, final={final:.4f})"
    )


def main() -> None:
    """Run both checks and print a summary so the script is usable standalone."""
    cfg = _load_cfg()
    device = torch.device("cpu")
    model = Markers2SMPL(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Smoke config: {cfg}")
    print(f"Model parameters: {n_params:,}")

    print("\n[1/2] Forward + gradient check ...", end=" ", flush=True)
    test_forward_shapes_and_finite_gradients()
    print("OK")

    print("[2/2] Overfit one batch (50 steps) ...", end=" ", flush=True)
    test_overfit_single_batch_reduces_loss()
    print("OK")

    # Re-run for visible numbers.
    torch.manual_seed(0)
    model = Markers2SMPL(cfg).to(device)
    batch = make_synthetic_batch(
        cfg["batch_size"], cfg["seq_len"],
        cfg["n_markers"], cfg["n_smpl_joints"], cfg["n_betas"], device,
    )
    opt = AdamW(model.parameters(), lr=cfg["lr"])
    kw = _loss_kwargs(cfg)

    print("\nstep   total     L_pose   L_shape   L_trans   L_smooth")
    for step in range(0, 51):
        losses = compute_losses(
            *model(batch["markers"], batch["mask"]),
            batch["poses_gt"], batch["betas_gt"], batch["trans_gt"], **kw,
        )
        if step % 10 == 0 or step == 50:
            print(
                f"{step:>4}  {losses['loss'].item():7.4f}  "
                f"{losses['L_pose'].item():7.4f}  {losses['L_shape'].item():7.4f}  "
                f"{losses['L_trans'].item():7.4f}  {losses['L_smooth'].item():7.4f}"
            )
        if step == 50:
            break
        opt.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
