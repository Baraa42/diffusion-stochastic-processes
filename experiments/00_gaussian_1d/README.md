# Stage 0: 1D Gaussian diffusion

Stage 0 is one controlled experiment: learn the score of a corrupted 1D
Gaussian with denoising score matching (DSM), compare it with the analytical
score, and use both scores in the same reverse Euler-Maruyama sampler.

## Forward process and score target

The data and corruption process are

$$
X_0\sim\mathcal N(\mu,\tau^2),\qquad
X_t=X_0+\sqrt t\,\varepsilon,\qquad
\varepsilon\sim\mathcal N(0,1),
$$

with `mu = 1`, `tau = 1`, and `t in [0.001, 2]`. Conditional on $X_0$,

$$
X_t\mid X_0\sim\mathcal N(X_0,t),\qquad
\nabla_{x_t}\log p(x_t\mid x_0)
=-\frac{x_t-x_0}{t}
=-\frac{\varepsilon}{\sqrt t}.
$$

This conditional corruption score is the DSM target. The network receives only
`(x_t, t)`—never `x0` or `eps`. Under squared error, the optimal prediction is
the conditional mean of the noisy target given `(x_t, t)`. The denoising-score
identity makes that conditional mean the marginal score. Here,

$$
X_t\sim\mathcal N(\mu,\tau^2+t),\qquad
s^*(x,t)=-\frac{x-\mu}{\tau^2+t}.
$$

The analytical score is used only for validation and the exact-score reverse
control, not as a training target.

## Time sampling and weighted DSM

Small time is statistically difficult because

$$
\mathrm{Var}\left(-\varepsilon/\sqrt t\right)=1/t.
$$

This run samples time log-uniformly,

$$
\log t\sim U(\log t_{\min},\log t_{\max}),
$$

which gives the small-time region more coverage than uniform time sampling. It
also changes the stochastic optimization behavior and did not improve every
fixed-time MSE in this seeded run. Training uses

$$
\mathbb E\left[t\left(s_\theta(X_t,t)+\frac{\varepsilon}{\sqrt t}\right)^2\right].
$$

Multiplication by $t$ counteracts the $1/t$ target variance without changing
the target itself or the population-optimal marginal score.

The model is a `2 -> 32 -> 32 -> 1` MLP with `Tanh` activations. It is trained
on CPU for 2,000 Adam steps with batch size 4,096, learning rate `1e-3`, and
seed 42.

## Score validation

Each fixed time uses a 300-point grid on `mu +/- 4*tau` and 100,000 fresh
on-distribution samples.

| t | Fixed-grid MSE | On-distribution MSE |
|---:|---:|---:|
| 0.001 | 1.764121e-01 | 4.015340e-03 |
| 0.005 | 1.707468e-01 | 3.792403e-03 |
| 0.010 | 1.638921e-01 | 3.603320e-03 |
| 0.050 | 1.172250e-01 | 2.529740e-03 |
| 0.200 | 2.874260e-02 | 5.994295e-04 |
| 1.000 | 5.082386e-04 | 2.299515e-04 |
| 2.000 | 1.138842e-03 | 6.919729e-04 |

The final logged weighted DSM loss was `0.885434`, and the fresh mixed-time
analytical MSE was `2.284013e-03`. Directional checks were positive left of the
mean, close to zero at the mean, and negative right of the mean. The larger
low-time grid errors occur mainly in the low-density far tails where the small
`Tanh` network saturates; the on-distribution errors are materially smaller.

## Reverse Euler-Maruyama validation

For the variance-exploding forward SDE used here, one backward step of size
`dt > 0` is

$$
X_{t-dt}\approx X_t+dt\,s(X_t,t)+\sqrt{dt}\,Z,
\qquad Z\sim\mathcal N(0,1).
$$

The score-agnostic sampler accepts either `exact_score` or the learned model.
Both runs start from the exact toy terminal marginal

$$
X_T\sim\mathcal N(\mu,\tau^2+t_{\max}).
$$

That initialization is available because this Gaussian experiment is
analytically controlled; it is not presented as a general practical terminal
prior. Within each paired comparison, exact and learned runs use the same
seeded `xT` and Brownian increments. The regular grid runs from `t_max` to
`t_min`, followed by one final step

$$
X_0\approx
X_{t_{\min}}
+
t_{\min}s(X_{t_{\min}},t_{\min})
+
\sqrt{t_{\min}}Z.
$$

This lands at data time zero while evaluating the score at exactly `t_min`,
never below the training range. The final target is
$X_0\sim\mathcal N(1,1)$, with mean `1.0` and variance `1.0`.

| N | Exact mean | Exact variance | Exact Euler expected variance | Learned mean | Learned variance |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.000963 | 1.008239 | 1.008929 | 1.007316 | 1.019939 |
| 500 | 1.000013 | 0.983055 | 1.001776 | 1.006140 | 0.994710 |
| 1000 | 1.005749 | 0.995526 | 1.000888 | 1.011906 | 1.007878 |
| 2000 | 1.005295 | 0.991503 | 1.000444 | 1.011426 | 1.003959 |

The analytically propagated exact Euler variance shows the discretization bias
decreasing from `8.95e-3` at 100 steps to `4.44e-4` at 2,000 steps. Sampled
exact variances are not monotone because 20,000-particle Monte Carlo error is
still visible. Comparing paired exact and learned samples isolates an
additional learned-score effect: learned variance stays roughly `0.012` above
the paired exact result. At 2,000 steps it is `1.003959`, a small positive bias
relative to the `1.0` target near the terminal low-time regime.

Reverse sampling materially contracts the initial variance from `3.0` toward
the target, but the result is not exact. The exact-score control diagnoses time
discretization and Monte Carlo error; the remaining paired gap diagnoses score
approximation error. Stage 0 implements no probability-flow ODE, flow matching,
or larger architecture.

## Reproduce

```bash
poetry install
poetry run pytest
poetry run python experiments/00_gaussian_1d/train.py
```

The run writes plots—including a final learned-sample histogram against the
analytical $\mathcal N(1,1)$ density—and machine-readable metrics to
`results/00_gaussian_1d/`.
