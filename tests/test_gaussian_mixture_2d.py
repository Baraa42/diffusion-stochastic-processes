"""Lightweight tests for Stage 1 Gaussian-mixture diffusion."""

import importlib.util
from pathlib import Path

import torch

experiment_path = (
    Path(__file__).parents[1]
    / "experiments"
    / "01_gaussian_mixture_2d"
    / "train.py"
)
spec = importlib.util.spec_from_file_location("gaussian_mixture_2d_train", experiment_path)
assert spec is not None and spec.loader is not None
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_sample_x0_shape() -> None:
    generator = torch.Generator().manual_seed(1)
    assert experiment.sample_x0(64, generator).shape == (64, 2)


def test_responsibilities_sum_to_one_for_batched_time() -> None:
    x = torch.randn(32, 2)
    t = torch.linspace(experiment.t_min, experiment.t_max, 32).reshape(-1, 1)
    responsibilities = experiment.mixture_responsibilities(x, t)
    assert responsibilities.shape == (32, 4)
    assert torch.allclose(
        responsibilities.sum(dim=1), torch.ones(32), atol=1e-6
    )


def test_exact_score_is_zero_at_symmetric_center() -> None:
    score = experiment.exact_score(torch.zeros(1, 2), 0.05)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-6)


def test_exact_score_is_nearly_zero_at_modes_for_small_time() -> None:
    score = experiment.exact_score(experiment.mus, experiment.t_min)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-5)


def test_exact_score_points_toward_nearby_mode() -> None:
    point = torch.tensor([[2.5, 2.0]])
    score = experiment.exact_score(point, 0.05)
    assert score[0, 0].item() < 0
    assert abs(score[0, 1].item()) < 1e-5


def test_reverse_sampler_preserves_shape() -> None:
    generator = torch.Generator().manual_seed(2)
    samples = experiment.reverse_sample(
        torch.zeros(32, 2), experiment.exact_score, 10, generator
    )
    assert samples.shape == (32, 2)


def test_learned_score_is_never_queried_below_t_min() -> None:
    queried_times: list[float] = []

    def recording_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        queried_times.append(t.min().item())
        return torch.zeros_like(x)

    generator = torch.Generator().manual_seed(3)
    experiment.reverse_sample(torch.zeros(16, 2), recording_score, 10, generator)
    assert abs(min(queried_times) - experiment.t_min) < 1e-8
    assert all(time >= experiment.t_min for time in queried_times)


def test_exact_reverse_sampling_recovers_mixture_modes() -> None:
    initial_generator = torch.Generator().manual_seed(4)
    xT = experiment.sample_xt(4_000, experiment.t_max, initial_generator)
    noise_generator = torch.Generator().manual_seed(5)
    samples = experiment.reverse_sample(
        xT, experiment.exact_score, 200, noise_generator
    )
    diagnostics = experiment.mode_diagnostics(samples)

    weights = torch.tensor(diagnostics["weights"])
    means = torch.tensor(diagnostics["means"])
    covariances = torch.tensor(diagnostics["covariances"])
    target_covariance = (experiment.sigma0**2) * torch.eye(2)

    assert torch.all(torch.abs(weights - 0.25) < 0.06)
    assert torch.all(torch.linalg.vector_norm(means - experiment.mus, dim=1) < 0.2)
    assert torch.all(torch.abs(covariances - target_covariance) < 0.08)
