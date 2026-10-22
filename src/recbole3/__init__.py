"""RecBole3.0 package."""

import os

# Keep Slurm/file logs readable: tqdm \r refreshes balloon .out files.
# Call sites may still pass disable=...; TQDM_DISABLE forces all tqdm bars off.
os.environ.setdefault("TQDM_DISABLE", "1")

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
