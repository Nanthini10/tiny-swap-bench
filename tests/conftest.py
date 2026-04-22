"""Pytest fixtures — deterministic cuDNN only under tests (project rule)."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cudnn_deterministic_for_tests():
    prev_d = torch.backends.cudnn.deterministic
    prev_b = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    torch.backends.cudnn.deterministic = prev_d
    torch.backends.cudnn.benchmark = prev_b
