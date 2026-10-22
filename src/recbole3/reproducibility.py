"""Run-level reproducibility helpers.

The shared :class:`~recbole3.pipeline.Pipeline` never seeded any RNG, so every
process launch drew a fresh weight initialization and a fresh train-shuffle
order (the latter also reshuffles the CF-align in-batch negatives). That made
single-run test metrics drift ~5-12% run-to-run. :func:`seed_everything` pins
python / numpy / torch / CUDA and, by default, requests deterministic CUDA
kernels so repeated runs of the same config match as closely as the hardware
allows.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed every RNG the training stack touches.

    Args:
        seed: Base seed for python, numpy, torch and all CUDA devices.
        deterministic: When set, also request deterministic cuDNN/cuBLAS
            kernels. Uses ``warn_only=True`` so an op without a deterministic
            implementation degrades to a warning instead of aborting the run.

    Returns:
        The (integer-normalized) seed actually applied.
    """

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS determinism must be requested before the first CUDA call; setting
    # it here works because seeding runs before any tensor reaches the GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:  # pragma: no cover - numpy is always present in practice
        pass

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch without the flag
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 so Ampere+ cards (A100/A5000/3090/L40S) do the same FP32
        # math as older cards (V100) -> maximizes cross-GPU-type consistency.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:  # pragma: no cover - older torch
            pass

    return seed


def seeded_generator(seed: int) -> "torch.Generator":
    """A CPU ``torch.Generator`` for DataLoader shuffling.

    Passing an explicit generator makes the shuffle order independent of how
    much global RNG earlier steps (e.g. weight init) consumed.
    """

    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` deriving per-worker seeds from torch's base seed."""

    import torch

    base_seed = torch.initial_seed() % (2**32)
    random.seed(base_seed + worker_id)
    try:
        import numpy as np

        np.random.seed((base_seed + worker_id) % (2**32))
    except ModuleNotFoundError:  # pragma: no cover
        pass
