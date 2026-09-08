# Stage 1 — 2D Gaussian mixture diffusion

This experiment extends the one-dimensional Gaussian validation to a genuinely
multimodal distribution while retaining an analytical score.

## Setup

The clean distribution is an equally weighted mixture of four isotropic
Gaussians:

$$
p_0(x)=\frac{1}{4}\sum_{k=1}^{4}
\mathcal N\!\left(x;\mu_k,\sigma_0^2 I_2\right),
\qquad \sigma_0=0.4,
$$

with means $(-2,-2)$, $(-2,2)$, $(2,-2)$, and $(2,2)$. The variance-exploding
forward process is

$$
X_t=X_0+\sqrt{t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I_2),
$$

so component $k$ at time $t$ has covariance $(\sigma_0^2+t)I_2$.

For equal mixture weights, the posterior responsibility and exact score are

$$
r_k(x,t)=\frac{p_{k,t}(x)}{\sum_j p_{j,t}(x)},
\qquad
s^*(x,t)=\sum_k r_k(x,t)
\left(-\frac{x-\mu_k}{\sigma_0^2+t}\right).
$$

The implementation evaluates log component densities and applies a softmax,
avoiding unstable probability-space normalization.

## Training

The score network is a `3 -> 64 -> 64 -> 2` multilayer perceptron with `Tanh`
activations. It receives $(x_1,x_2,t)$ and is trained only against the
conditional DSM target

$$
Y=-\frac{\varepsilon}{\sqrt{t}}.
$$

Times are sampled log-uniformly on $[10^{-3},2]$. The optimized objective is

$$
L=\mathbb E\left[t\left\|s_\theta(X_t,t)-Y\right\|^2\right].
$$

The seeded CPU run uses batch size 4096, 3000 Adam steps, and learning rate
$10^{-3}$. Its final weighted DSM loss is `1.4351`, while its fresh mixed-time
analytical score MSE is `0.07556`. The nonzero DSM loss is expected: the
conditional target retains randomness after conditioning on $(X_t,t)$.

## Score validation

The coordinate-averaged on-distribution MSE from 100,000 fresh samples per time
is:

| $t$ | MSE |
|---:|---:|
| 0.001 | 0.176191 |
| 0.010 | 0.124827 |
| 0.050 | 0.039731 |
| 0.100 | 0.036358 |
| 0.200 | 0.033962 |
| 0.500 | 0.007492 |
| 1.000 | 0.004316 |
| 2.000 | 0.003506 |

Uniform-grid error is deliberately reported separately because it weights
low-density regions much more heavily:

| $t$ | mean error norm | median | maximum |
|---:|---:|---:|---:|
| 0.001 | 3.4416 | 3.1554 | 9.5603 |
| 0.050 | 1.7749 | 1.5268 | 6.5678 |
| 0.200 | 0.5249 | 0.2907 | 2.8354 |
| 1.000 | 0.0892 | 0.0797 | 0.2778 |

Exact and learned quiver fields are visually similar, especially at moderate
and high times. At low time, the learned score is reasonable near a mode but
does not exactly preserve symmetry at the center or inter-mode boundaries, and
it underestimates the far-tail magnitude. The directions remain qualitatively
correct. This explains why on-distribution error can be modest while the error
on a uniform spatial grid is much larger.

The forward-corruption plot uses a shared clean sample set and shows four tight
modes broadening, overlapping, and becoming strongly blurred as $t$ increases.

## Reverse generation

Because the forward SDE is $dX_t=dW_t$, reverse Euler–Maruyama uses

$$
X_{t-\Delta t}\approx X_t+\Delta t\,s_t(X_t)+\sqrt{\Delta t}\,Z.
$$

The generic sampler accepts either the exact score or the learned network. It
integrates from $t_{\max}$ to $t_{\min}$ and takes one final
$t_{\min}\rightarrow0$ step by evaluating the score exactly at $t_{\min}$; the
network is never queried below its training range. Both comparisons start from
the same samples and use the same Brownian random stream.

For this controlled analytical experiment, terminal samples are drawn directly
from the exact mixture $p_{t_{\max}}$. This is an analytical control, not a
general practical diffusion initialization strategy.

Nearest-mean assignment shows that exact-score reverse sampling reproduces the
four target modes, weights, means, and covariance $0.16I_2$. Learned-score
sampling also recovers all four modes without missing or spurious modes. Its
weights and means are accurate, while within-mode variances are slightly high.
The fraction farther than $3\sigma_0=1.2$ from every mode is `1.20%` for the
exact sampler and `2.58%` for the learned sampler, compared with the theoretical
$e^{-9/2}=1.11\%$.

The model therefore learned the main multimodal score geometry. Its principal
weakness is low-time, low-density space. Imperfect low-time score magnitude is
consistent with the observed slight over-dispersion, but these diagnostics do
not establish strict causality.

Full seeded measurements, including strategic-point checks and per-mode
covariances, are stored in
[`metrics.json`](../../results/01_gaussian_mixture_2d/metrics.json).

## Run

```bash
poetry run python experiments/01_gaussian_mixture_2d/train.py
poetry run pytest
```

This stage does not implement Stage 2, flow matching, or a probability-flow ODE.
