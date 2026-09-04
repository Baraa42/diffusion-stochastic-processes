"""Minimal checks for the initial project environment."""

import torch


def test_torch_is_available() -> None:
    assert torch.__version__
