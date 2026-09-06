# Denoising score matching

## Conditional corruption score

For

$$
X_t\mid X_0=x_0\sim\mathcal N(x_0,t),
$$

the conditional log-density has derivative

$$
\nabla_{x_t}\log p(x_t\mid x_0)
=-\frac{x_t-x_0}{t}
=-\frac{\varepsilon}{\sqrt t}.
$$

This target is available during training because the procedure samples $x_0$,
$t$, and $\varepsilon$, then constructs $x_t=x_0+\sqrt t\,\varepsilon$. The
network itself receives only $(x_t,t)$; it never receives $x_0$ or
$\varepsilon$.

## Why the conditional target learns the marginal score

The corrupted marginal density is

$$
p_t(x)=\int p(x\mid x_0)\,p_{\mathrm{data}}(x_0)\,dx_0.
$$

Differentiate under the integral and use
$\nabla_x p(x\mid x_0)=p(x\mid x_0)\nabla_x\log p(x\mid x_0)$:

$$
\nabla_x p_t(x)
=
\int
p(x\mid x_0)\,
\nabla_x\log p(x\mid x_0)\,
p_{\mathrm{data}}(x_0)\,dx_0.
$$

Bayes' rule gives the posterior over clean samples:

$$
p(x_0\mid x)
=
\frac{p(x\mid x_0)p_{\mathrm{data}}(x_0)}{p_t(x)}.
$$

Dividing the differentiated marginal by $p_t(x)$ and substituting this
posterior yields

$$
\begin{aligned}
\nabla_x\log p_t(x)
&=
\frac{\nabla_xp_t(x)}{p_t(x)} \\
&=
\int
\nabla_x\log p(x\mid x_0)\,
p(x_0\mid x)\,dx_0 \\
&=
\mathbb E\left[
\nabla_x\log p(X_t\mid X_0)
\mid X_t=x
\right].
\end{aligned}
$$

Define the noisy regression target

$$
Y=-\frac{\varepsilon}{\sqrt t}.
$$

For squared-error regression, conditioning on the input gives

$$
\mathbb E\left[(s_\theta-Y)^2\mid X_t=x,t\right]
=
\left(s_\theta-\mathbb E[Y\mid X_t=x,t]\right)^2
+
\mathrm{Var}(Y\mid X_t=x,t).
$$

The second term cannot be reduced by the network. The first is minimized by

$$
s^*(x,t)=\mathbb E[Y\mid X_t=x,t].
$$

Using the denoising-score identity above,

$$
s^*(x,t)=\nabla_x\log p_t(x).
$$

Thus DSM trains on a tractable conditional score while learning the marginal
score.

## Irreducible noise and small time

The conditional target is random even for a fixed network input $(x_t,t)$.
Consequently, the DSM training loss has an irreducible positive component and
need not approach zero even when the marginal score is learned accurately.

Before conditioning, its variance is

$$
\mathrm{Var}\left(-\varepsilon/\sqrt t\right)=\frac{1}{t}.
$$

The target therefore becomes increasingly noisy as $t$ approaches zero.
Stage 0 uses the weighted objective

$$
\mathbb E\left[t(s_\theta-Y)^2\right].
$$

At any fixed positive $t$, multiplication by the positive scalar $t$ does not
change the population minimizer. Across sampled times, however, it changes
their relative contribution to stochastic gradients and therefore changes
optimization behavior.

Time is sampled log-uniformly:

$$
\log t\sim U(\log t_{\min},\log t_{\max}).
$$

This increases low-time coverage relative to uniform sampling. It is not a
universal solution: it changes time emphasis and gradient noise, and it did
not improve every fixed-time MSE in the seeded Stage 0 experiment.
