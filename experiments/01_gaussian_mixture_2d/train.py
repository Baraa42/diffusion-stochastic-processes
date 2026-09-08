"""Stage 1: learn and reverse-sample a four-component 2D Gaussian mixture."""

import json
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

plt.switch_backend("Agg")


sigma0 = 0.4
mus = torch.tensor(
    [
        [-2.0, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.0, 2.0],
    ]
)
t_min = 1e-3
t_max = 2.0

seed = 42
batch_size = 4096
training_steps = 3000
learning_rate = 1e-3
log_every = 100

t_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
field_t_values = [0.001, 0.05, 0.2, 1.0]
strategic_t_values = [0.001, 0.05]
strategic_points = torch.tensor(
    [
        [2.0, 2.0],
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [4.0, 4.0],
    ]
)
score_validation_samples = 100_000
reverse_sample_count = 20_000
reverse_steps = 1000
reverse_seed = 31415
device = torch.device("cpu")

results_dir = (
    Path(__file__).resolve().parents[2] / "results" / "01_gaussian_mixture_2d"
)
ScoreFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sample_x0(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample n points from the equally weighted clean mixture."""
    component = torch.randint(0, len(mus), (n,), generator=generator, device=device)
    noise = torch.randn(n, 2, generator=generator, device=device)
    return mus[component] + sigma0 * noise


def sample_xt(
    n: int, t_value: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample directly from the mixture marginal at one fixed time."""
    component = torch.randint(0, len(mus), (n,), generator=generator, device=device)
    scale = math.sqrt(sigma0**2 + t_value)
    noise = torch.randn(n, 2, generator=generator, device=device)
    return mus[component] + scale * noise


def sample_batch(
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x0, t, eps, xt, and target for weighted DSM training."""
    x0 = sample_x0(n)
    log_t = math.log(t_min) + (math.log(t_max) - math.log(t_min)) * torch.rand(
        n, 1, device=device
    )
    t = torch.exp(log_t)
    eps = torch.randn(n, 2, device=device)
    xt = x0 + torch.sqrt(t) * eps
    target = -eps / torch.sqrt(t)
    return x0, t, eps, xt, target


def _mixture_variance(x: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    """Return sigma0^2 + t with a broadcastable [1 or N, 1] shape."""
    t_tensor = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.reshape(1, 1)
    elif t_tensor.ndim == 1:
        t_tensor = t_tensor.reshape(-1, 1)
    elif t_tensor.ndim != 2 or t_tensor.shape[1] != 1:
        raise ValueError("t must be scalar, [1], [N], or [N, 1]")
    if t_tensor.shape[0] not in (1, x.shape[0]):
        raise ValueError("batched t must have the same leading dimension as x")
    return sigma0**2 + t_tensor


def mixture_responsibilities(
    x: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """Return posterior component probabilities with shape [N, 4]."""
    variance = _mixture_variance(x, t)
    differences = x[:, None, :] - mus[None, :, :]
    squared_distances = differences.square().sum(dim=2)
    log_probabilities = -squared_distances / (2 * variance)
    return torch.softmax(log_probabilities, dim=1)


def exact_score(x: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    """Return the analytical score of the corrupted Gaussian mixture."""
    variance = _mixture_variance(x, t)
    differences = x[:, None, :] - mus[None, :, :]
    component_scores = -differences / variance.unsqueeze(2)
    responsibilities = mixture_responsibilities(x, t)
    return (responsibilities.unsqueeze(2) * component_scores).sum(dim=1)


class ScoreNetwork(nn.Module):
    """A 3 -> 64 -> 64 -> 2 score network."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([x, t], dim=1))


@torch.no_grad()
def analytical_validation_mse(model: nn.Module, n: int = 20_000) -> float:
    """Return coordinate-averaged exact-score MSE on a fresh mixed-time batch."""
    _, t, _, xt, _ = sample_batch(n)
    return torch.mean((model(xt, t) - exact_score(xt, t)) ** 2).item()


def train() -> tuple[ScoreNetwork, dict[str, list[float]]]:
    """Train only on the weighted conditional DSM objective."""
    torch.manual_seed(seed)
    model = ScoreNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: dict[str, list[float]] = {
        "step": [],
        "weighted_dsm_loss": [],
        "analytical_mse": [],
    }

    model.train()
    for step in range(1, training_steps + 1):
        _, t, _, xt, target = sample_batch(batch_size)
        prediction = model(xt, t)
        squared_error = (prediction - target).square().sum(dim=1, keepdim=True)
        loss = torch.mean(t * squared_error)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0:
            model.eval()
            validation_mse = analytical_validation_mse(model)
            model.train()
            history["step"].append(step)
            history["weighted_dsm_loss"].append(loss.item())
            history["analytical_mse"].append(validation_mse)
            print(
                f"step={step:4d}  weighted_dsm_loss={loss.item():.6f}  "
                f"analytical_mse={validation_mse:.6e}"
            )

    return model, history


@torch.no_grad()
def fixed_t_score_validation(model: nn.Module) -> list[dict[str, float]]:
    """Measure on-distribution exact-score MSE at fixed times."""
    model.eval()
    metrics: list[dict[str, float]] = []
    print("\nFixed-t on-distribution score MSE")
    for t_value in t_values:
        xt = sample_xt(score_validation_samples, t_value)
        t = torch.full((score_validation_samples, 1), t_value, device=device)
        mse = torch.mean((model(xt, t) - exact_score(xt, t)) ** 2).item()
        metrics.append({"t": t_value, "coordinate_mse": mse})
        print(f"t={t_value:>5g}  coordinate_mse={mse:.6e}")
    return metrics


def score_grid(points_per_axis: int = 25) -> torch.Tensor:
    """Return a shared flat [-4, 4]^2 grid."""
    coordinates = torch.linspace(-4.0, 4.0, points_per_axis)
    grid_x, grid_y = torch.meshgrid(coordinates, coordinates, indexing="xy")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    return points


def draw_score_field(
    axis: plt.Axes,
    points: torch.Tensor,
    scores: torch.Tensor,
    title: str,
) -> None:
    """Draw one score field with a common spatial and arrow scale."""
    magnitudes = torch.linalg.vector_norm(scores, dim=1)
    axis.quiver(
        points[:, 0].numpy(),
        points[:, 1].numpy(),
        scores[:, 0].numpy(),
        scores[:, 1].numpy(),
        magnitudes.numpy(),
        angles="xy",
        scale_units="xy",
        scale=10,
        cmap="viridis",
        width=0.004,
    )
    axis.scatter(mus[:, 0].numpy(), mus[:, 1].numpy(), marker="x", color="red")
    axis.set_xlim(-4.2, 4.2)
    axis.set_ylim(-4.2, 4.2)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.grid(alpha=0.2)


@torch.no_grad()
def score_field_diagnostics(
    model: nn.Module,
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    """Save quiver fields and report uniform-grid and strategic-point errors."""
    model.eval()
    points = score_grid()
    grid_metrics: list[dict[str, float]] = []

    exact_figure, exact_axes = plt.subplots(2, 2, figsize=(11, 10))
    comparison_figure, comparison_axes = plt.subplots(
        len(field_t_values), 2, figsize=(12, 20)
    )
    print("\nUniform-grid score error norm")
    for index, t_value in enumerate(field_t_values):
        t = torch.full((len(points), 1), t_value, device=device)
        exact = exact_score(points, t)
        learned = model(points, t)
        error = torch.linalg.vector_norm(learned - exact, dim=1)
        row = {
            "t": t_value,
            "mean_error_norm": error.mean().item(),
            "median_error_norm": error.median().item(),
            "max_error_norm": error.max().item(),
        }
        grid_metrics.append(row)
        print(
            f"t={t_value:>5g}  mean={row['mean_error_norm']:.6f}  "
            f"median={row['median_error_norm']:.6f}  max={row['max_error_norm']:.6f}"
        )

        draw_score_field(
            exact_axes.flat[index], points, exact, f"Exact score, t={t_value:g}"
        )
        draw_score_field(
            comparison_axes[index, 0], points, exact, f"Exact, t={t_value:g}"
        )
        draw_score_field(
            comparison_axes[index, 1], points, learned, f"Learned, t={t_value:g}"
        )

    exact_figure.suptitle("Exact Gaussian-mixture score fields")
    exact_figure.tight_layout()
    exact_figure.savefig(results_dir / "exact_score_fields.png", dpi=160)
    plt.close(exact_figure)

    comparison_figure.suptitle("Exact vs learned Gaussian-mixture score fields")
    comparison_figure.tight_layout()
    comparison_figure.savefig(
        results_dir / "exact_vs_learned_score_fields.png", dpi=160
    )
    plt.close(comparison_figure)

    strategic_metrics: list[dict[str, object]] = []
    print("\nStrategic score checks")
    for t_value in strategic_t_values:
        t = torch.full((len(strategic_points), 1), t_value, device=device)
        exact = exact_score(strategic_points, t)
        learned = model(strategic_points, t)
        for point, exact_value, learned_value in zip(
            strategic_points, exact, learned, strict=True
        ):
            error_norm = torch.linalg.vector_norm(learned_value - exact_value).item()
            strategic_metrics.append(
                {
                    "t": t_value,
                    "point": point.tolist(),
                    "exact_score": exact_value.tolist(),
                    "learned_score": learned_value.tolist(),
                    "error_norm": error_norm,
                }
            )
            print(
                f"t={t_value:g} point={point.tolist()} exact={exact_value.tolist()} "
                f"learned={learned_value.tolist()} error={error_norm:.6f}"
            )

    return grid_metrics, strategic_metrics


def plot_training_metrics(history: dict[str, list[float]]) -> None:
    """Save weighted DSM and analytical validation histories."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["weighted_dsm_loss"])
    axes[0].set_title("Weighted DSM training loss")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("weighted squared error")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(history["step"], history["analytical_mse"])
    axes[1].set_title("MSE against exact mixture score")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("coordinate MSE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(results_dir / "training_metrics.png", dpi=160)
    plt.close(figure)


def plot_forward_corruption() -> None:
    """Show the same clean sample/noise pair at four corruption times."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    x0 = sample_x0(5_000, generator)
    eps = torch.randn(5_000, 2, generator=generator, device=device)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for axis, t_value in zip(axes, [0.0, 0.1, 0.5, 2.0], strict=True):
        xt = x0 + math.sqrt(t_value) * eps
        axis.scatter(xt[:, 0].numpy(), xt[:, 1].numpy(), s=3, alpha=0.35)
        axis.set_title(f"t = {t_value:g}")
        axis.set_xlim(-6, 6)
        axis.set_ylim(-6, 6)
        axis.set_aspect("equal")
        axis.set_xlabel("x1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("x2")
    figure.suptitle("Forward corruption of the four-component mixture")
    figure.tight_layout()
    figure.savefig(results_dir / "forward_corruption.png", dpi=160)
    plt.close(figure)


@torch.no_grad()
def reverse_sample(
    xT: torch.Tensor,
    score_fn: ScoreFunction,
    n_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-Maruyama reverse sampling to zero without querying below t_min."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    x = xT.clone()
    dt = (t_max - t_min) / n_steps
    for step in range(n_steps):
        time = t_max - step * dt
        t = torch.full((len(x), 1), time, dtype=x.dtype, device=x.device)
        noise = torch.randn(x.shape, generator=generator, device=x.device)
        x = x + dt * score_fn(x, t) + math.sqrt(dt) * noise

    t = torch.full((len(x), 1), t_min, dtype=x.dtype, device=x.device)
    noise = torch.randn(x.shape, generator=generator, device=x.device)
    return x + t_min * score_fn(x, t) + math.sqrt(t_min) * noise


def assign_modes(samples: torch.Tensor) -> torch.Tensor:
    """Assign each point to its nearest clean mixture mean."""
    squared_distances = (samples[:, None, :] - mus[None, :, :]).square().sum(dim=2)
    return squared_distances.argmin(dim=1)


def mode_diagnostics(samples: torch.Tensor) -> dict[str, list[object]]:
    """Return nearest-mode weights, empirical means, and empirical covariances."""
    assignment = assign_modes(samples)
    weights: list[float] = []
    means: list[list[float]] = []
    covariances: list[list[list[float]]] = []
    for component in range(len(mus)):
        component_samples = samples[assignment == component]
        mean = component_samples.mean(dim=0)
        centered = component_samples - mean
        covariance = centered.T @ centered / (len(component_samples) - 1)
        weights.append(len(component_samples) / len(samples))
        means.append(mean.tolist())
        covariances.append(covariance.tolist())
    return {"weights": weights, "means": means, "covariances": covariances}


def off_mode_fraction(samples: torch.Tensor) -> float:
    """Return the fraction farther than 3*sigma0 from its nearest mode."""
    distances = torch.linalg.vector_norm(
        samples[:, None, :] - mus[None, :, :], dim=2
    )
    return (distances.min(dim=1).values > 3 * sigma0).float().mean().item()


def print_mode_diagnostics(name: str, diagnostics: dict[str, list[object]]) -> None:
    """Print compact per-mode reverse statistics."""
    print(f"\n{name} reverse mode diagnostics")
    for index in range(len(mus)):
        print(
            f"mode={index} weight={diagnostics['weights'][index]:.6f} "
            f"mean={diagnostics['means'][index]} "
            f"covariance={diagnostics['covariances'][index]}"
        )


@torch.no_grad()
def reverse_generation(
    model: nn.Module,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, list[object]],
    dict[str, list[object]],
    float,
    float,
]:
    """Run paired exact/learned reverse sampling and summarize modes."""
    model.eval()
    initial_generator = torch.Generator(device="cpu").manual_seed(reverse_seed)
    xT = sample_xt(reverse_sample_count, t_max, initial_generator)
    exact_generator = torch.Generator(device="cpu").manual_seed(reverse_seed + 1)
    learned_generator = torch.Generator(device="cpu").manual_seed(reverse_seed + 1)
    exact_samples = reverse_sample(xT, exact_score, reverse_steps, exact_generator)
    learned_samples = reverse_sample(xT, model, reverse_steps, learned_generator)
    true_generator = torch.Generator(device="cpu").manual_seed(reverse_seed + 2)
    true_samples = sample_x0(reverse_sample_count, true_generator)

    exact_diagnostics = mode_diagnostics(exact_samples)
    learned_diagnostics = mode_diagnostics(learned_samples)
    exact_off_mode = off_mode_fraction(exact_samples)
    learned_off_mode = off_mode_fraction(learned_samples)
    print_mode_diagnostics("Exact-score", exact_diagnostics)
    print_mode_diagnostics("Learned-score", learned_diagnostics)
    print(
        f"\noff_mode_fraction exact={exact_off_mode:.6f} "
        f"learned={learned_off_mode:.6f}"
    )
    return (
        true_samples,
        exact_samples,
        learned_samples,
        exact_diagnostics,
        learned_diagnostics,
        exact_off_mode,
        learned_off_mode,
    )


def plot_reverse_samples(
    true_samples: torch.Tensor,
    exact_samples: torch.Tensor,
    learned_samples: torch.Tensor,
) -> None:
    """Save true, exact-score, and learned-score samples side by side."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    panels = [
        ("True X0 samples", true_samples),
        ("Exact-score reverse", exact_samples),
        ("Learned-score reverse", learned_samples),
    ]
    for axis, (title, samples) in zip(axes, panels, strict=True):
        shown = samples[:5_000]
        axis.scatter(shown[:, 0].numpy(), shown[:, 1].numpy(), s=3, alpha=0.35)
        axis.scatter(mus[:, 0].numpy(), mus[:, 1].numpy(), marker="x", color="red")
        axis.set_title(title)
        axis.set_xlim(-4.5, 4.5)
        axis.set_ylim(-4.5, 4.5)
        axis.set_aspect("equal")
        axis.set_xlabel("x1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("x2")
    figure.tight_layout()
    figure.savefig(results_dir / "true_vs_exact_vs_learned_samples.png", dpi=160)
    plt.close(figure)


def plot_mode_weights(
    exact_diagnostics: dict[str, list[object]],
    learned_diagnostics: dict[str, list[object]],
) -> None:
    """Compare exact and learned generated mode weights with the target."""
    positions = torch.arange(len(mus), dtype=torch.float32)
    width = 0.25
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        (positions - width).numpy(),
        exact_diagnostics["weights"],
        width=width,
        label="exact score",
    )
    axis.bar(
        positions.numpy(),
        learned_diagnostics["weights"],
        width=width,
        label="learned score",
    )
    axis.bar(
        (positions + width).numpy(),
        [0.25] * len(mus),
        width=width,
        label="target",
    )
    axis.set_xticks(positions.numpy(), [str(index) for index in range(len(mus))])
    axis.set_xlabel("nearest mode")
    axis.set_ylabel("empirical weight")
    axis.set_title("Generated mixture weights")
    axis.set_ylim(0, 0.3)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "mode_weights.png", dpi=160)
    plt.close(figure)


def save_metrics(
    history: dict[str, list[float]],
    fixed_t_metrics: list[dict[str, float]],
    grid_metrics: list[dict[str, float]],
    strategic_metrics: list[dict[str, object]],
    exact_diagnostics: dict[str, list[object]],
    learned_diagnostics: dict[str, list[object]],
    exact_off_mode: float,
    learned_off_mode: float,
) -> None:
    """Save all current seeded-run diagnostics as JSON."""
    metrics = {
        "configuration": {
            "sigma0": sigma0,
            "mus": mus.tolist(),
            "mixture_weights": [0.25] * len(mus),
            "t_min": t_min,
            "t_max": t_max,
            "time_sampling": "log_uniform",
            "loss": "mean(t * sum((prediction - target)^2, coordinates))",
            "seed": seed,
            "batch_size": batch_size,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "device": str(device),
            "score_validation_samples": score_validation_samples,
            "reverse_sample_count": reverse_sample_count,
            "reverse_steps": reverse_steps,
            "reverse_seed": reverse_seed,
            "reverse_endpoint": 0.0,
        },
        "training_summary": {
            "final_weighted_dsm_loss": history["weighted_dsm_loss"][-1],
            "final_analytical_mse": history["analytical_mse"][-1],
            "history": history,
        },
        "fixed_t_score_mse": fixed_t_metrics,
        "uniform_grid_error": grid_metrics,
        "strategic_point_diagnostics": strategic_metrics,
        "reverse_target": {
            "weights": [0.25] * len(mus),
            "means": mus.tolist(),
            "covariance": [[sigma0**2, 0.0], [0.0, sigma0**2]],
            "off_mode_radius": 3 * sigma0,
            "theoretical_off_mode_fraction": math.exp(-9 / 2),
        },
        "exact_reverse_diagnostics": {
            **exact_diagnostics,
            "off_mode_fraction": exact_off_mode,
        },
        "learned_reverse_diagnostics": {
            **learned_diagnostics,
            "off_mode_fraction": learned_off_mode,
        },
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} seed={seed} batch_size={batch_size} "
        f"training_steps={training_steps}"
    )
    model, history = train()
    plot_training_metrics(history)
    plot_forward_corruption()
    fixed_t_metrics = fixed_t_score_validation(model)
    grid_metrics, strategic_metrics = score_field_diagnostics(model)
    (
        true_samples,
        exact_samples,
        learned_samples,
        exact_diagnostics,
        learned_diagnostics,
        exact_off_mode,
        learned_off_mode,
    ) = reverse_generation(model)
    plot_reverse_samples(true_samples, exact_samples, learned_samples)
    plot_mode_weights(exact_diagnostics, learned_diagnostics)
    save_metrics(
        history,
        fixed_t_metrics,
        grid_metrics,
        strategic_metrics,
        exact_diagnostics,
        learned_diagnostics,
        exact_off_mode,
        learned_off_mode,
    )
    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()
