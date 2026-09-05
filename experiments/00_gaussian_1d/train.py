"""Train a tiny network with DSM and validate it against an exact 1D score."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

plt.switch_backend("Agg")


# X_0 ~ N(mu, tau^2), and X_t = X_0 + sqrt(t) * eps.
mu = 1.0
tau = 1.0
t_min = 0.05
t_max = 2.0

seed = 7
batch_size = 4096
training_steps = 2000
learning_rate = 1e-3
log_every = 100

t_values = [0.05, 0.2, 0.5, 1.0, 2.0]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
results_dir = Path(__file__).resolve().parents[2] / "results" / "00_gaussian_1d"


def sample_batch(batch_size: int):
    """Return x0, t, eps, xt, and the DSM target, each shaped [batch_size, 1]."""
    x0 = mu + tau * torch.randn(batch_size, 1, device=device)
    t = t_min + (t_max - t_min) * torch.rand(batch_size, 1, device=device)
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
        inputs = torch.cat([x, t], dim=1)
        center = inputs.new_tensor([mu, (t_min + t_max) / 2])
        scale = inputs.new_tensor([4 * tau, (t_max - t_min) / 2])
        return self.layers((inputs - center) / scale)


@torch.no_grad()
def analytical_validation_mse(model: nn.Module, size: int = 20_000) -> float:
    """Compare the network with the exact score on a fresh forward-process batch."""
    _, t, _, xt, _ = sample_batch(size)
    prediction = model(xt, t)
    return nn.functional.mse_loss(prediction, exact_score(xt, t)).item()


def train() -> tuple[ScoreNetwork, list[int], list[float], list[float]]:
    """Fit only to noisy conditional DSM targets."""
    torch.manual_seed(seed)
    model = ScoreNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    logged_steps: list[int] = []
    dsm_losses: list[float] = []
    validation_mses: list[float] = []

    model.train()
    for step in range(1, training_steps + 1):
        _, t, _, xt, target = sample_batch(batch_size)
        prediction = model(xt, t)
        dsm_loss = loss_function(prediction, target)

        optimizer.zero_grad()
        dsm_loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0:
            model.eval()
            validation_mse = analytical_validation_mse(model)
            model.train()

            logged_steps.append(step)
            dsm_losses.append(dsm_loss.item())
            validation_mses.append(validation_mse)
            print(
                f"step={step:4d}  dsm_loss={dsm_loss.item():.6f}  "
                f"analytical_mse={validation_mse:.6e}"
            )

    return model, logged_steps, dsm_losses, validation_mses


@torch.no_grad()
def validate(model: nn.Module) -> None:
    """Run fixed-grid, on-distribution, and directional validation."""
    model.eval()
    results_dir.mkdir(parents=True, exist_ok=True)

    x_grid = torch.linspace(mu - 4 * tau, mu + 4 * tau, 300, device=device).reshape(-1, 1)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)

    print("\nFixed-grid and on-distribution analytical score MSE")
    for axis, t_value in zip(axes.flat, t_values, strict=False):
        t_grid = torch.full_like(x_grid, t_value)
        learned = model(x_grid, t_grid)
        exact = exact_score(x_grid, t_grid)
        grid_mse = nn.functional.mse_loss(learned, exact).item()

        xt = mu + (tau**2 + t_value) ** 0.5 * torch.randn(100_000, 1, device=device)
        t = torch.full_like(xt, t_value)
        on_distribution_mse = nn.functional.mse_loss(model(xt, t), exact_score(xt, t)).item()
        print(
            f"t={t_value:>4.2f}  grid_mse={grid_mse:.6e}  "
            f"on_distribution_mse={on_distribution_mse:.6e}"
        )

        axis.plot(x_grid.cpu().numpy(), exact.cpu().numpy(), label="exact", linewidth=2)
        axis.plot(x_grid.cpu().numpy(), learned.cpu().numpy(), "--", label="learned", linewidth=2)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.axvline(mu, color="black", linewidth=0.5)
        axis.set_title(f"t = {t_value:g}")
        axis.set_xlabel("x")
        axis.set_ylabel("score")
        axis.grid(alpha=0.25)

    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    figure.suptitle("Learned DSM score vs exact Gaussian marginal score")
    figure.tight_layout()
    figure.savefig(results_dir / "learned_vs_exact_scores.png", dpi=160)
    plt.close(figure)

    points = torch.tensor([[mu - 2], [mu], [mu + 2]], device=device)
    print("\nDirectional sanity check (learned / exact)")
    print("       " + "  ".join(f"t={value:g}" for value in t_values))
    for point in points:
        x = point.reshape(1, 1).expand(len(t_values), 1)
        t = torch.tensor(t_values, device=device).reshape(-1, 1)
        learned = model(x, t).squeeze(1).cpu().tolist()
        exact = exact_score(x, t).squeeze(1).cpu().tolist()
        values = "  ".join(f"{left:+.4f}/{right:+.4f}" for left, right in zip(learned, exact, strict=True))
        print(f"x={point.item():+.1f}  {values}")


def plot_training_metrics(steps: list[int], dsm_losses: list[float], validation_mses: list[float]) -> None:
    """Save DSM loss and exact-score validation MSE without interactive display."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, dsm_losses)
    axes[0].set_title("DSM training loss")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)

    axes[1].semilogy(steps, validation_mses)
    axes[1].set_title("MSE against exact marginal score")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(results_dir / "training_metrics.png", dpi=160)
    plt.close(figure)


def main() -> None:
    print(f"device={device}  seed={seed}  batch_size={batch_size}  steps={training_steps}")
    model, steps, dsm_losses, validation_mses = train()
    plot_training_metrics(steps, dsm_losses, validation_mses)
    validate(model)
    print(f"\nSaved plots to {results_dir}")


if __name__ == "__main__":
    main()
