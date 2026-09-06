# Gaussian diffusion

## Data distribution and forward process

Let the one-dimensional data distribution be

$$
X_0\sim\mathcal N(\mu,\tau^2).
$$

Stage 0 uses the variance-exploding forward SDE

$$
dX_t=dW_t.
$$

Integrating from zero to $t$ gives

$$
X_t=X_0+W_t.
$$

Because $W_t\sim\mathcal N(0,t)$, the same process can be parameterized as

$$
X_t=X_0+\sqrt t\,\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,1).
$$

Conditioned on one clean value,

$$
X_t\mid X_0=x_0\sim\mathcal N(x_0,t).
$$

The sum of independent Gaussian variables is Gaussian, so marginalizing over
$X_0$ gives

$$
X_t\sim\mathcal N(\mu,\tau^2+t).
$$

## Exact marginal score

The log-density, up to terms independent of $x$, is

$$
\log p_t(x)=-\frac{(x-\mu)^2}{2(\tau^2+t)}+\text{constant}.
$$

Therefore,

$$
\nabla_x\log p_t(x)=-\frac{x-\mu}{\tau^2+t}.
$$

The score is positive to the left of $\mu$, negative to the right, and zero at
$\mu$, so it points toward the Gaussian mean. As $t$ increases, the variance
$\tau^2+t$ grows and the score magnitude decreases at any fixed displacement
from the mean.

## Meaning of time

Here $t$ is **artificial diffusion/noising time**: it indexes how much Gaussian
noise has been added to a static data sample. It is not necessarily the
physical or process time of a system whose state genuinely evolves. Later
stochastic-process experiments may use time as part of the data itself; that
data/process time must be kept distinct from the artificial diffusion time used
to corrupt samples for score learning.
