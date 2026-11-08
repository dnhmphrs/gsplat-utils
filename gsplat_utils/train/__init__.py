"""Gaussian-splat training on top of `gsplat` (CUDA only).

Nothing here imports torch at module level: `gsplat_utils.train.dataset` loads
COLMAP data with numpy/PIL, and `gsplat_utils.train.trainer.train()` imports
torch/gsplat lazily so the rest of the package runs on CPU-only machines.
"""
from .dataset import ColmapData, load_colmap  # noqa: F401
from .trainer import TrainConfig, train  # noqa: F401
