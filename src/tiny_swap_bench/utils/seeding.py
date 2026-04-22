"""Deterministic RNG seeding (training: no cudnn deterministic for speed)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, cudnn_deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = not cudnn_deterministic
