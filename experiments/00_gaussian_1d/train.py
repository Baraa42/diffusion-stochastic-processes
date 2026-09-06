"""Complete Stage 0: DSM score learning and reverse validation for a 1D Gaussian."""

import json
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

plt.switch_backend("Agg")


# X_0 ~ N(mu, tau^2), and X_t = X_0 + sqrt(t) * eps.
mu = 1.0
tau = 1.0
t_min = 1e-3
t_max = 2.0

seed = 42
batch_size = 4096
training_steps = 2000
learning_rate = 1e-3
log_every = 100

t_values = [0.001, 0.005, 0.01, 0.05, 0.2, 1.0, 2.0]
n_values = [100, 500, 1000, 2000]
score_validation_samples = 100_000
reverse_sample_count = 20_000
reverse_seed = 31415

# CPU makes this small controlled experiment reproducible across common machines.
device = torch.device("cpu")
results_dir = Path(__file__).resolve().parents[2] / "results" / "00_gaussian_1d"

ScoreFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sample_batch(batch_size: int):
    """Return x0, t, eps, xt, and target, each with shape [batch_size, 1]."""
    x0 = mu + tau * torch.randn(batch_size, 1, device=device)
    log_t = math.log(t_min) + (math.log(t_max) - math.log(t_min)) * torch.rand(
        batch_size, 1, device=device
    )
    t = torch.exp(log_t)
    eps = torch.randn(batch_size, 1, device=device)
    xt = x0 + torch.sqrt(t) * eps
    target = -eps / torch.sqrt(t)
    return x0, t, eps, xt, target


def exact_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return the marginal score of X_t ~ N(mu, tau^2 + t)."""
    return -(x - mu) / (tau**2 + t)


class ScoreNetwork(nn.Module):
    """A 2 -> 32 -> 32 -> 1 MLP for s_theta(x, t)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([x, t], dim=1))


@torch.no_grad()
def analytical_validation_mse(model: nn.Module, size: int = 20_000) -> float:
    """Compare the model with the exact score on a fresh forward-process batch."""
    _, t, _, xt, _ = sample_batch(size)
    return nn.functional.mse_loss(model(xt, t), exact_score(xt, t)).item()


def train() -> tuple[ScoreNetwork, dict[str, list[float]]]:
    """Train only against the weighted, noisy conditional DSM objective."""
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
        loss = torch.mean(t * (prediction - target) ** 2)

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
def fixed_t_score_validation(model: nn.Module) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    """Measure fixed-grid/on-distribution error and check score direction."""
    model.eval()
    results_dir.mkdir(parents=True, exist_ok=True)
    x_grid = torch.linspace(mu - 4 * tau, mu + 4 * tau, 300, device=device).reshape(-1, 1)
    figure, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
    fixed_time_metrics: list[dict[str, float]] = []

    print("\nFixed-t analytical score MSE")
    for axis, t_value in zip(axes.flat, t_values, strict=False):
        t_grid = torch.full_like(x_grid, t_value)
        learned = model(x_grid, t_grid)
        exact = exact_score(x_grid, t_grid)
        grid_mse = nn.functional.mse_loss(learned, exact).item()

        xt = mu + (tau**2 + t_value) ** 0.5 * torch.randn(
            score_validation_samples, 1, device=device
        )
        t = torch.full_like(xt, t_value)
        on_distribution_mse = nn.functional.mse_loss(
            model(xt, t), exact_score(xt, t)
        ).item()
        fixed_time_metrics.append(
            {
                "t": t_value,
                "grid_mse": grid_mse,
                "on_distribution_mse": on_distribution_mse,
            }
        )
        print(
            f"t={t_value:>5g}  grid_mse={grid_mse:.6e}  "
            f"on_distribution_mse={on_distribution_mse:.6e}"
        )

        axis.plot(x_grid.numpy(), exact.numpy(), label="exact", linewidth=2)
        axis.plot(x_grid.numpy(), learned.numpy(), "--", label="learned", linewidth=2)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.axvline(mu, color="black", linewidth=0.5)
        axis.set_title(f"t = {t_value:g}")
        axis.set_xlabel("x")
        axis.set_ylabel("score")
        axis.grid(alpha=0.25)

    axes.flat[0].legend()
    for axis in axes.flat[len(t_values) :]:
        axis.axis("off")
    figure.suptitle("Learned weighted-DSM score vs exact Gaussian score")
    figure.tight_layout()
    figure.savefig(results_dir / "learned_vs_exact_scores.png", dpi=160)
    plt.close(figure)

    points = torch.tensor([[mu - 2], [mu], [mu + 2]], device=device)
    directional_results: list[dict[str, object]] = []
    print("\nDirectional sanity check (learned / exact)")
    print("       " + "  ".join(f"t={value:g}" for value in t_values))
    for point in points:
        x = point.reshape(1, 1).expand(len(t_values), 1)
        t = torch.tensor(t_values, device=device).reshape(-1, 1)
        learned = model(x, t).squeeze(1).tolist()
        exact = exact_score(x, t).squeeze(1).tolist()
        directional_results.append(
            {"x": point.item(), "learned": learned, "exact": exact}
        )
        values = "  ".join(
            f"{left:+.4f}/{right:+.4f}"
            for left, right in zip(learned, exact, strict=True)
        )
        print(f"x={point.item():+.1f}  {values}")

    return fixed_time_metrics, directional_results


@torch.no_grad()
def reverse_sample(
    xT: torch.Tensor,
    score_fn: ScoreFunction,
    n_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-Maruyama sampling from t_max down to t_min using any score_fn."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    x = xT.clone()
    dt = (t_max - t_min) / n_steps
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps):
        time = t_max - step * dt
        t = torch.full_like(x, time)
        noise = torch.randn(x.shape, generator=generator, device=x.device)
        x = x + dt * score_fn(x, t) + sqrt_dt * noise
    return x


def exact_euler_variance(n_steps: int) -> float:
    """Propagate the exact variance of the Euler scheme without Monte Carlo noise."""
    dt = (t_max - t_min) / n_steps
    variance = tau**2 + t_max
    for step in range(n_steps):
        time = t_max - step * dt
        contraction = 1 - dt / (tau**2 + time)
        variance = contraction**2 * variance + dt
    return variance


@torch.no_grad()
def reverse_discretization_experiment(model: nn.Module) -> list[dict[str, float]]:
    """Compare exact and learned scores with paired initial data and noise."""
    model.eval()
    initial_generator = torch.Generator(device="cpu").manual_seed(reverse_seed)
    xT = mu + (tau**2 + t_max) ** 0.5 * torch.randn(
        reverse_sample_count, 1, generator=initial_generator, device=device
    )

    reverse_results: list[dict[str, float]] = []
    print("\nReverse Euler-Maruyama discretization")
    for n_steps in n_values:
        noise_seed = reverse_seed + n_steps
        exact_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        learned_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        exact_samples = reverse_sample(xT, exact_score, n_steps, exact_generator)
        learned_samples = reverse_sample(xT, model, n_steps, learned_generator)

        result = {
            "n_steps": n_steps,
            "exact_mean": exact_samples.mean().item(),
            "exact_variance": exact_samples.var(unbiased=True).item(),
            "exact_euler_variance": exact_euler_variance(n_steps),
            "learned_mean": learned_samples.mean().item(),
            "learned_variance": learned_samples.var(unbiased=True).item(),
        }
        reverse_results.append(result)
        print(
            f"N={n_steps:4d}  exact=({result['exact_mean']:.6f}, "
            f"{result['exact_variance']:.6f})  learned=({result['learned_mean']:.6f}, "
            f"{result['learned_variance']:.6f})  "
            f"exact_euler_variance={result['exact_euler_variance']:.6f}"
        )

    return reverse_results


def plot_training_metrics(history: dict[str, list[float]]) -> None:
    """Save weighted DSM loss and exact-score validation MSE."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["weighted_dsm_loss"])
    axes[0].set_title("Weighted DSM training loss")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("weighted MSE")
    axes[0].grid(alpha=0.25)

    axes[1].semilogy(history["step"], history["analytical_mse"])
    axes[1].set_title("MSE against exact marginal score")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(results_dir / "training_metrics.png", dpi=160)
    plt.close(figure)


def plot_reverse_metrics(reverse_results: list[dict[str, float]]) -> None:
    """Save reverse mean, variance, and variance-error diagnostics."""
    steps = [row["n_steps"] for row in reverse_results]
    exact_means = [row["exact_mean"] for row in reverse_results]
    learned_means = [row["learned_mean"] for row in reverse_results]
    exact_variances = [row["exact_variance"] for row in reverse_results]
    exact_euler_variances = [row["exact_euler_variance"] for row in reverse_results]
    learned_variances = [row["learned_variance"] for row in reverse_results]
    target_variance = tau**2 + t_min

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, exact_means, "o-", label="exact score")
    axis.plot(steps, learned_means, "o-", label="learned score")
    axis.axhline(mu, color="black", linestyle="--", label="target")
    axis.set_xscale("log")
    axis.set_xlabel("Euler steps")
    axis.set_ylabel("recovered mean")
    axis.set_title("Reverse discretization: mean")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "reverse_discretization_mean.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, exact_variances, "o-", label="exact score samples")
    axis.plot(steps, exact_euler_variances, "o--", label="exact Euler expectation")
    axis.plot(steps, learned_variances, "o-", label="learned score samples")
    axis.axhline(target_variance, color="black", linestyle="--", label="target")
    axis.set_xscale("log")
    axis.set_xlabel("Euler steps")
    axis.set_ylabel("recovered variance")
    axis.set_title("Reverse discretization: variance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "reverse_discretization_variance.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.loglog(
        steps,
        [abs(value - target_variance) for value in exact_variances],
        "o-",
        label="exact score samples",
    )
    axis.loglog(
        steps,
        [abs(value - target_variance) for value in exact_euler_variances],
        "o--",
        label="exact Euler discretization",
    )
    axis.loglog(
        steps,
        [abs(value - target_variance) for value in learned_variances],
        "o-",
        label="learned score samples",
    )
    axis.set_xlabel("Euler steps")
    axis.set_ylabel("absolute variance error")
    axis.set_title("Reverse variance error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "reverse_variance_error.png", dpi=160)
    plt.close(figure)


def save_metrics(
    history: dict[str, list[float]],
    fixed_time_metrics: list[dict[str, float]],
    directional_results: list[dict[str, object]],
    reverse_results: list[dict[str, float]],
) -> None:
    """Write every reported metric from this seeded run to JSON."""
    metrics = {
        "configuration": {
            "mu": mu,
            "tau": tau,
            "t_min": t_min,
            "t_max": t_max,
            "time_sampling": "log_uniform",
            "loss": "mean(t * (prediction - target) ** 2)",
            "seed": seed,
            "batch_size": batch_size,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "device": str(device),
            "score_validation_samples": score_validation_samples,
            "reverse_sample_count": reverse_sample_count,
            "reverse_seed": reverse_seed,
            "n_values": n_values,
        },
        "training_summary": {
            "final_weighted_dsm_loss": history["weighted_dsm_loss"][-1],
            "final_analytical_mse": history["analytical_mse"][-1],
            "history": history,
        },
        "fixed_time_score_metrics": fixed_time_metrics,
        "directional_check": directional_results,
        "reverse_target": {"mean": mu, "variance": tau**2 + t_min},
        "reverse_results": reverse_results,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  seed={seed}  batch_size={batch_size}  steps={training_steps}")
    model, history = train()
    plot_training_metrics(history)
    fixed_time_metrics, directional_results = fixed_t_score_validation(model)
    reverse_results = reverse_discretization_experiment(model)
    plot_reverse_metrics(reverse_results)
    save_metrics(history, fixed_time_metrics, directional_results, reverse_results)
    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()
