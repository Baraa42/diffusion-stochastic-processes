"""Lightweight checks for the 1D Gaussian Stage 0 implementation."""

import importlib.util
from pathlib import Path

import torch

experiment_path = Path(__file__).parents[1] / "experiments" / "00_gaussian_1d" / "train.py"
spec = importlib.util.spec_from_file_location("gaussian_1d_train", experiment_path)
assert spec is not None and spec.loader is not None
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_torch_is_available() -> None:
    assert torch.__version__


def test_sample_batch_shapes_and_time_range() -> None:
    tensors = experiment.sample_batch(64)
    assert all(tensor.shape == (64, 1) for tensor in tensors)
    t = tensors[1]
    assert torch.all(t >= experiment.t_min)
    assert torch.all(t <= experiment.t_max)


def test_exact_score_is_zero_at_mean() -> None:
    x = torch.tensor([[experiment.mu]], device=experiment.device)
    t = torch.tensor([[0.5]], device=experiment.device)
    assert torch.equal(experiment.exact_score(x, t), torch.zeros_like(x))


def test_exact_score_points_toward_mean() -> None:
    x = torch.tensor(
        [[experiment.mu - 1.0], [experiment.mu + 1.0]], device=experiment.device
    )
    t = torch.full_like(x, 0.5)
    score = experiment.exact_score(x, t)
    assert score[0].item() > 0
    assert score[1].item() < 0


def test_reverse_sampler_preserves_shape() -> None:
    xT = torch.zeros(32, 1)
    generator = torch.Generator().manual_seed(1)
    samples = experiment.reverse_sample(xT, experiment.exact_score, 10, generator)
    assert samples.shape == xT.shape


def test_reverse_sampler_never_queries_below_t_min() -> None:
    queried_times: list[float] = []

    def recording_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        queried_times.append(t.min().item())
        return experiment.exact_score(x, t)

    generator = torch.Generator().manual_seed(4)
    experiment.reverse_sample(torch.zeros(16, 1), recording_score, 10, generator)
    assert abs(min(queried_times) - experiment.t_min) < 1e-8
    assert all(time >= experiment.t_min for time in queried_times)


def test_exact_reverse_sampler_recovers_target_moments() -> None:
    generator = torch.Generator().manual_seed(2)
    xT = experiment.mu + (experiment.tau**2 + experiment.t_max) ** 0.5 * torch.randn(
        4_000, 1, generator=generator
    )
    noise_generator = torch.Generator().manual_seed(3)
    samples = experiment.reverse_sample(
        xT, experiment.exact_score, 200, noise_generator
    )
    target_variance = experiment.tau**2
    assert abs(samples.mean().item() - experiment.mu) < 0.08
    assert abs(samples.var(unbiased=True).item() - target_variance) < 0.12
