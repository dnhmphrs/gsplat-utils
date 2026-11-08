"""Train a 3D Gaussian Splat with `gsplat` from the COLMAP output of `gsu sfm`.

GPU only: gsplat has no CPU rasteriser. Everything torch/gsplat is imported
inside functions so the rest of gsplat-utils works without them.

The optimisation follows gsplat's reference `simple_trainer.py`: Adam per
parameter group, exponential decay on the means' learning rate, L1 + SSIM
loss, `DefaultStrategy` (ADC densification) or `MCMCStrategy`. The world frame
is never normalised or re-centred: the PLY is written in COLMAP coordinates so
`gsu align`, `gsu geometry` and `gsu web` can use the same cameras.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common import log, write_json
from ..export.ply import SH_C0, Gaussians, write_gaussian_ply
from ..paths import Workspace
from .dataset import ColmapData, load_colmap, resolve_sparse_dir, to_torch


@dataclass
class TrainConfig:
    # -- what the CLI exposes --------------------------------------------------
    iters: int = 30000
    strategy: str = "default"            # "default" (ADC densification) | "mcmc"
    max_gaussians: int = 2_000_000       # hard cap for mcmc, growth stop for default
    sh_degree: int = 3
    downscale: int = 1                   # train on 1/N resolution
    data: str = "undistorted"            # "undistorted" (PINHOLE) | "sparse" (needs PINHOLE model)
    eval_every: int = 0                  # hold out every N-th image for PSNR/SSIM (0 = none)
    save_every: int = 10000              # checkpoint (+ eval) period
    resume: Path | None = None           # checkpoint to resume from
    device: str = "cuda"
    # -- loss ----------------------------------------------------------------------
    ssim_lambda: float = 0.2
    opacity_reg: float | None = None     # None -> 0.01 for mcmc, 0 for default
    scale_reg: float | None = None       # None -> 0.01 for mcmc, 0 for default
    random_bkgd: bool = False
    # -- learning rates (means lr is multiplied by the scene scale) ----------------
    lr_means: float = 1.6e-4
    lr_means_final_ratio: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    # -- initialisation ------------------------------------------------------------
    init_opacity: float | None = None    # None -> 0.1 (default) / 0.5 (mcmc)
    init_scale: float | None = None      # None -> 1.0 (default) / 0.1 (mcmc)
    init_max_points: int = 1_000_000
    # -- densification schedule ----------------------------------------------------
    sh_degree_interval: int = 1000
    refine_start_iter: int = 500
    refine_stop_iter: int | None = None  # None -> iters//2 (default) / iters - iters//6 (mcmc)
    refine_every: int = 100
    reset_every: int = 3000
    # -- rasteriser ----------------------------------------------------------------
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    absgrad: bool = False
    antialiased: bool = False
    # -- bookkeeping ---------------------------------------------------------------
    batch_size: int = 1                  # kept for completeness; this trainer renders one view per step
    seed: int = 42
    log_every: int = 100
    ply_every: int = 0                   # also write model_<step>.ply every N steps (0 = off)
    eval_images_saved: int = 4           # side-by-side PNGs written per eval

    def resolved(self) -> "TrainConfig":
        """Fill the strategy-dependent `None` defaults."""
        c = TrainConfig(**asdict(self))
        mcmc = c.strategy == "mcmc"
        if c.opacity_reg is None:
            c.opacity_reg = 0.01 if mcmc else 0.0
        if c.scale_reg is None:
            c.scale_reg = 0.01 if mcmc else 0.0
        if c.init_opacity is None:
            c.init_opacity = 0.5 if mcmc else 0.1
        if c.init_scale is None:
            c.init_scale = 0.1 if mcmc else 1.0
        if c.refine_stop_iter is None:
            c.refine_stop_iter = (c.iters - c.iters // 6) if mcmc else c.iters // 2
        if c.resume is not None:
            c.resume = Path(c.resume)
        return c

    def to_json(self) -> dict:
        d = asdict(self)
        d["resume"] = None if self.resume is None else str(self.resume)
        return d


# ---------------------------------------------------------------------------
# data location
# ---------------------------------------------------------------------------
def data_dirs(ws: Workspace, data: str) -> tuple[Path, Path]:
    """(sparse model dir, image dir) for the requested data set."""
    if data == "undistorted":
        sparse = ws.undistorted / "sparse"
        images = ws.undistorted / "images"
        if not images.exists():
            raise FileNotFoundError(
                f"{ws.undistorted} not found. Run `gsu sfm` (it writes the undistorted set) "
                f"or train with --data sparse on a PINHOLE model."
            )
        return resolve_sparse_dir(sparse), images
    if data == "sparse":
        return resolve_sparse_dir(ws.model), ws.frames
    raise ValueError(f"unknown data set {data!r} (undistorted|sparse)")


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def init_gaussians_numpy(points: np.ndarray, rgb: np.ndarray, sh_degree: int, init_opacity: float,
                         init_scale: float, max_points: int | None, seed: int) -> dict[str, np.ndarray]:
    """3DGS-style initialisation from SfM points (numpy; testable without torch)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float64)
    col = np.asarray(rgb, dtype=np.float64)
    if col.max() > 1.0:
        col = col / 255.0
    if max_points and len(pts) > max_points:
        sel = rng.choice(len(pts), max_points, replace=False)
        pts, col = pts[sel], col[sel]
    n = len(pts)
    if n == 0:
        raise RuntimeError("no 3D points to initialise from")
    k = min(4, n)
    d, _ = cKDTree(pts).query(pts, k=k)
    if k > 1:
        dist_avg = np.sqrt(np.mean(d[:, 1:] ** 2, axis=1))
    else:
        dist_avg = np.ones(n)
    dist_avg = np.maximum(dist_avg, 1e-7)
    scales = np.log(dist_avg * init_scale)[:, None].repeat(3, axis=1)
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opac = np.full(n, math.log(init_opacity / (1.0 - init_opacity)))
    K = (sh_degree + 1) ** 2
    sh0 = ((col - 0.5) / SH_C0)[:, None, :]          # (N,1,3)
    shN = np.zeros((n, K - 1, 3), dtype=np.float64)  # (N,K-1,3)
    return {
        "means": pts.astype(np.float32), "scales": scales.astype(np.float32),
        "quats": quats.astype(np.float32), "opacities": opac.astype(np.float32),
        "sh0": sh0.astype(np.float32), "shN": shN.astype(np.float32),
    }


def _make_params(arrays: dict[str, Any], device):
    import torch

    return torch.nn.ParameterDict({
        k: torch.nn.Parameter(torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v).float().to(device))
        for k, v in arrays.items()
    })


def _make_optimizers(params, cfg: TrainConfig, scene_scale: float):
    import torch

    lrs = {
        "means": cfg.lr_means * scene_scale, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k], "name": k}], eps=1e-15)
        for k in params.keys()
    }
    gamma = cfg.lr_means_final_ratio ** (1.0 / max(1, cfg.iters))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=gamma)
    return optimizers, scheduler


def _make_strategy(cfg: TrainConfig, scene_scale: float):
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    if cfg.strategy == "mcmc":
        strategy = MCMCStrategy(
            cap_max=cfg.max_gaussians, refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter, refine_every=cfg.refine_every, verbose=True,
        )
        state = strategy.initialize_state()
    elif cfg.strategy == "default":
        strategy = DefaultStrategy(
            verbose=True, absgrad=cfg.absgrad, refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter, reset_every=cfg.reset_every, refine_every=cfg.refine_every,
        )
        state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        raise ValueError(f"unknown strategy {cfg.strategy!r} (default|mcmc)")
    return strategy, state


def params_to_gaussians(params) -> Gaussians:
    import torch

    with torch.no_grad():
        q = params["quats"].detach()
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return Gaussians(
            means=params["means"].detach().cpu().numpy().astype(np.float32),
            scales=params["scales"].detach().cpu().numpy().astype(np.float32),
            quats=q.cpu().numpy().astype(np.float32),
            opacities=params["opacities"].detach().cpu().numpy().astype(np.float32),
            sh0=params["sh0"].detach()[:, 0, :].cpu().numpy().astype(np.float32),
            shN=params["shN"].detach().cpu().numpy().astype(np.float32),
        )


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def _render(params, viewmat, K, width, height, sh_degree, cfg: TrainConfig, absgrad: bool):
    import torch
    from gsplat import rasterization

    return rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.cat([params["sh0"], params["shN"]], 1),
        viewmats=viewmat[None],
        Ks=K[None],
        width=int(width),
        height=int(height),
        sh_degree=int(sh_degree),
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        packed=cfg.packed,
        absgrad=absgrad,
        rasterize_mode="antialiased" if cfg.antialiased else "classic",
        render_mode="RGB",
    )


def evaluate(params, tensors: dict, val_idx: np.ndarray, cfg: TrainConfig, step: int,
             out_dir: Path | None = None) -> dict:
    """PSNR / SSIM on the held-out views (full SH degree). Writes a few GT|render PNGs."""
    import torch
    from PIL import Image

    from .losses import psnr, ssim

    psnrs, ssims = [], []
    saved = 0
    with torch.no_grad():
        for i in val_idx:
            i = int(i)
            w, h = tensors["sizes"][i]
            pixels = tensors["images"][i].to(params["means"].device, non_blocking=True).float() / 255.0
            renders, alphas, _ = _render(params, tensors["w2c"][i], tensors["Ks"][i], w, h, cfg.sh_degree, cfg, False)
            render = renders[0].clamp(0, 1)
            psnrs.append(float(psnr(render, pixels)))
            ssims.append(float(ssim(render.permute(2, 0, 1)[None], pixels.permute(2, 0, 1)[None])))
            if out_dir is not None and saved < cfg.eval_images_saved:
                out_dir.mkdir(parents=True, exist_ok=True)
                side = torch.cat([pixels, render], dim=1).mul(255).round().byte().cpu().numpy()
                Image.fromarray(side).save(out_dir / f"step{step:06d}_view{i:04d}.png")
                saved += 1
    res = {
        "step": int(step), "n_views": len(psnrs),
        "psnr": float(np.mean(psnrs)) if psnrs else None,
        "ssim": float(np.mean(ssims)) if ssims else None,
    }
    log.info("eval @ %d: PSNR %.2f  SSIM %.4f  (%d views)", step, res["psnr"] or 0.0, res["ssim"] or 0.0, len(psnrs))
    return res


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, step: int, params, optimizers, scheduler, strategy_state, cfg: TrainConfig,
                    scene_scale: float) -> None:
    import torch

    torch.save({
        "step": int(step),
        "splats": {k: v.detach().cpu() for k, v in params.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "scheduler": scheduler.state_dict(),
        "strategy_state": strategy_state,
        "cfg": cfg.to_json(),
        "scene_scale": float(scene_scale),
    }, path)
    log.info("checkpoint %s (%d gaussians)", path, len(params["means"]))


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def train(ws: Workspace, cfg: TrainConfig) -> Path:
    cfg = cfg.resolved()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit("torch is not installed. On a CUDA machine: pip install 'gsplat-utils[train]'") from exc
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "Training needs an NVIDIA GPU: gsplat has no CPU rasteriser and torch reports no CUDA device. "
            "Copy the work folder (frames/, colmap/) to a GPU machine, `pip install 'gsplat-utils[train]'`, "
            "and run `gsu train --work <dir>` there. The CPU stages (align/geometry) do not need the splat."
        )
    try:
        import gsplat  # noqa: F401  (JIT-compiles its CUDA kernels on first use)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("gsplat is not installed: pip install gsplat  (needs CUDA + torch)") from exc

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ws.splat.mkdir(parents=True, exist_ok=True)

    sparse_dir, images_dir = data_dirs(ws, cfg.data)
    data: ColmapData = load_colmap(sparse_dir, images_dir, downscale=cfg.downscale, eval_every=cfg.eval_every,
                                   require_pinhole=True)
    scene_scale = 1.1 * data.scene_scale
    tensors = to_torch(data, device=str(device))
    train_idx = data.train_idx if len(data.train_idx) else np.arange(data.n_images)

    log_path = ws.splat / "train_log.json"
    log_data: dict[str, Any] = {
        "config": cfg.to_json(), "data": {"sparse": str(sparse_dir), "images": str(images_dir),
                                            "n_train": int(len(train_idx)), "n_val": int(len(data.val_idx)),
                                            "scene_scale": scene_scale, "camera_model": data.camera_model},
        "steps": [], "eval": [], "final": None,
    }

    # -- parameters / optimisers / strategy ------------------------------------------
    start_step = 0
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        params = _make_params(ckpt["splats"], device)
        optimizers, scheduler = _make_optimizers(params, cfg, ckpt.get("scene_scale", scene_scale))
        for k, o in optimizers.items():
            if k in ckpt["optimizers"]:
                o.load_state_dict(ckpt["optimizers"][k])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        strategy, strategy_state = _make_strategy(cfg, scene_scale)
        if ckpt.get("strategy_state"):
            strategy_state.update(ckpt["strategy_state"])
        start_step = int(ckpt["step"])
        log.info("resumed %s at step %d with %d gaussians", cfg.resume, start_step, len(params["means"]))
    else:
        arrays = init_gaussians_numpy(data.points, data.points_rgb, cfg.sh_degree, cfg.init_opacity,
                                      cfg.init_scale, cfg.init_max_points, cfg.seed)
        params = _make_params(arrays, device)
        optimizers, scheduler = _make_optimizers(params, cfg, scene_scale)
        strategy, strategy_state = _make_strategy(cfg, scene_scale)
        log.info("initialised %d gaussians from SfM points (sh degree %d)", len(params["means"]), cfg.sh_degree)
    strategy.check_sanity(params, optimizers)
    is_default = cfg.strategy == "default"

    from .losses import ssim

    t0 = time.time()
    last_log = t0
    for step in range(start_step, cfg.iters):
        i = int(train_idx[rng.integers(len(train_idx))])
        w, h = tensors["sizes"][i]
        pixels = tensors["images"][i].to(device, non_blocking=True).float().div_(255.0)[None]  # (1,H,W,3)
        sh_deg = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        renders, alphas, info = _render(params, tensors["w2c"][i], tensors["Ks"][i], w, h, sh_deg, cfg,
                                        absgrad=cfg.absgrad and is_default)
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            renders = renders + bkgd * (1.0 - alphas)

        strategy.step_pre_backward(params=params, optimizers=optimizers, state=strategy_state, step=step, info=info)

        l1 = (renders - pixels).abs().mean()
        ssim_val = ssim(renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = l1 * (1.0 - cfg.ssim_lambda) + (1.0 - ssim_val) * cfg.ssim_lambda
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * torch.exp(params["scales"]).abs().mean()
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        # densify / prune (after the optimiser step, as in gsplat's simple_trainer)
        if is_default:
            over_cap = len(params["means"]) >= cfg.max_gaussians
            saved_thr = None
            if over_cap:  # keep pruning, stop growing
                saved_thr = strategy.grow_grad2d
                strategy.grow_grad2d = float("inf")
            try:
                strategy.step_post_backward(params=params, optimizers=optimizers, state=strategy_state,
                                            step=step, info=info, packed=cfg.packed)
            finally:
                if saved_thr is not None:
                    strategy.grow_grad2d = saved_thr
        else:
            strategy.step_post_backward(params=params, optimizers=optimizers, state=strategy_state,
                                        step=step, info=info, lr=scheduler.get_last_lr()[0])

        # -- bookkeeping ---------------------------------------------------------------
        if (step + 1) % cfg.log_every == 0 or step == start_step:
            now = time.time()
            n = len(params["means"])
            done = step + 1 - start_step
            rate = done / max(now - t0, 1e-6)
            eta = (cfg.iters - step - 1) / max(rate, 1e-6)
            log.info("step %6d/%d  loss %.4f  l1 %.4f  ssim %.4f  gaussians %8d  %.1f it/s  eta %dm%02ds",
                     step + 1, cfg.iters, float(loss), float(l1), float(ssim_val), n, rate, int(eta // 60), int(eta % 60))
            log_data["steps"].append({
                "step": step + 1, "loss": float(loss), "l1": float(l1), "ssim": float(ssim_val),
                "n_gaussians": int(n), "elapsed_s": now - t0, "lr_means": scheduler.get_last_lr()[0],
                "sh_degree": sh_deg,
            })
            if now - last_log > 30:
                write_json(log_path, log_data)
                last_log = now
        final = step + 1 == cfg.iters
        if cfg.save_every and ((step + 1) % cfg.save_every == 0 or final):
            save_checkpoint(ws.splat / f"ckpt_{step + 1}.pt", step + 1, params, optimizers, scheduler,
                            strategy_state, cfg, scene_scale)
            if len(data.val_idx):
                log_data["eval"].append(evaluate(params, tensors, data.val_idx, cfg, step + 1, ws.splat / "eval"))
            write_json(log_path, log_data)
        if cfg.ply_every and (step + 1) % cfg.ply_every == 0 and not final:
            write_gaussian_ply(ws.splat / f"model_{step + 1}.ply", params_to_gaussians(params))

    # -- final export (COLMAP frame) ------------------------------------------------------
    g = params_to_gaussians(params)
    write_gaussian_ply(ws.splat_ply, g)
    meta = {"frame": "colmap", "n": int(len(g)), "iters": int(cfg.iters), "sh_degree": int(cfg.sh_degree),
            "data": cfg.data, "downscale": int(cfg.downscale), "strategy": cfg.strategy,
            "train_time_s": time.time() - t0}
    write_json(ws.splat / "model.ply.json", meta)
    log_data["final"] = meta
    if len(data.val_idx) and (not log_data["eval"] or log_data["eval"][-1]["step"] != cfg.iters):
        log_data["eval"].append(evaluate(params, tensors, data.val_idx, cfg, cfg.iters, ws.splat / "eval"))
    write_json(log_path, log_data)
    log.info("wrote %s (%d gaussians, %.1f min)", ws.splat_ply, len(g), (time.time() - t0) / 60)
    print(ws.splat_ply)
    return ws.splat_ply
