# Stage 0: 1D Gaussian DSM validation

This experiment uses

\[
X_0 \sim \mathcal N(1, 1), \qquad
X_t = X_0 + \sqrt{t}\,\varepsilon, \qquad
\varepsilon \sim \mathcal N(0,1),
\]

with time sampled uniformly from `[0.05, 2.0]`. The network is trained only
against the conditional DSM target

\[
-\frac{\varepsilon}{\sqrt t}.
\]

The analytical marginal score is reserved for validation:

\[
s^*(x,t)=-\frac{x-1}{1+t}.
\]

## Model and training

The score model receives only `(x_t, t)`. It concatenates and applies fixed
input centering/scaling before a `2 -> 32 -> 32 -> 1` MLP with `Tanh`
activations. Training uses Adam with learning rate `1e-3`, batch size `4096`,
seed `7`, and `2000` steps.

## Validation

The learned and exact scores are compared at
`t = [0.05, 0.2, 0.5, 1.0, 2.0]` on a fixed 300-point grid and on 100,000
fresh samples from the exact marginal at each time. The run also checks the
score at `x = mu - 2`, `mu`, and `mu + 2`.

Observed MSE from the seeded run:

| t | Fixed-grid MSE | On-distribution MSE |
|---:|---:|---:|
| 0.05 | 9.480102e-02 | 3.651623e-03 |
| 0.20 | 1.953202e-02 | 6.024865e-04 |
| 0.50 | 5.642961e-04 | 4.511883e-04 |
| 1.00 | 4.443171e-04 | 2.069998e-04 |
| 2.00 | 4.238228e-04 | 5.589580e-04 |

At step 2000, the sampled DSM loss was `1.310870` and the fresh-batch MSE
against the analytical score was `3.542737e-04`. The DSM loss fluctuated around
its expected positive floor (approximately `1.35`) rather than approaching
zero. This is expected: even after conditioning on `(x_t, t)`, the conditional
target `-eps / sqrt(t)` retains randomness. Its conditional mean is the smooth
marginal score learned by the MSE regressor.

The directional checks had the correct signs on both sides of the mean, were
near zero at the mean, and decreased in magnitude with increasing time. The
larger small-time fixed-grid error occurs mainly near the low-density
`mu +/- 4*tau` endpoints, where a small `Tanh` network has approximation error;
the on-distribution error is much smaller.

Run with:

```bash
poetry run python experiments/00_gaussian_1d/train.py
```

The script saves plots under `results/00_gaussian_1d/`. Reverse diffusion,
reverse SDE sampling, and probability-flow ODEs are not implemented.
