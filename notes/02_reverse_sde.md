# Reverse SDE for the 1D Gaussian

## Exact one-step conditional

Consider one small forward transition,

$$
X_{t+dt}\mid X_t=x\sim\mathcal N(x,dt),
$$

with the marginal at time $t$

$$
X_t\sim\mathcal N(\mu,V_t),
\qquad
V_t=\tau^2+t.
$$

Write $y=X_{t+dt}$. Bayes' rule gives

$$
p(x\mid y)\propto p(y\mid x)p_t(x).
$$

The likelihood and prior contribute the quadratic terms

$$
-\frac{(y-x)^2}{2dt}
-\frac{(x-\mu)^2}{2V_t}.
$$

Completing the square, or adding Gaussian precisions, gives

$$
\begin{aligned}
\mathrm{Var}(X_t\mid X_{t+dt}=y)
&=
\left(\frac{1}{V_t}+\frac{1}{dt}\right)^{-1}
=\frac{V_tdt}{V_t+dt},\\
\mathbb E[X_t\mid X_{t+dt}=y]
&=
\frac{V_ty+dt\,\mu}{V_t+dt}.
\end{aligned}
$$

Therefore,

$$
X_t\mid X_{t+dt}=y
\sim
\mathcal N\left(
\frac{V_ty+dt\,\mu}{V_t+dt},
\frac{V_tdt}{V_t+dt}
\right).
$$

## Rewriting the reverse mean with the score

At time $t+dt$, the marginal variance is $V_t+dt$, so

$$
s_{t+dt}(y)=-\frac{y-\mu}{V_t+dt}.
$$

Then

$$
\begin{aligned}
y+dt\,s_{t+dt}(y)
&=
y-dt\,\frac{y-\mu}{V_t+dt}\\
&=
\frac{V_ty+dt\,\mu}{V_t+dt}\\
&=
\mathbb E[X_t\mid X_{t+dt}=y].
\end{aligned}
$$

Also,

$$
\frac{V_tdt}{V_t+dt}=dt+O(dt^2).
$$

The small-step reverse Euler-Maruyama update is therefore

$$
X_t
\approx
X_{t+dt}
+
dt\,s_{t+dt}(X_{t+dt})
+
\sqrt{dt}\,Z,
\qquad
Z\sim\mathcal N(0,1).
$$

In practice the unknown score is replaced by the learned approximation

$$
s_\theta(x,t).
$$

## Stage 0 implementation and controls

The regular grid integrates from $t_{\max}$ to the smallest trained time
$t_{\min}$. One final Euler step reaches data time zero:

$$
X_0
\approx
X_{t_{\min}}
+
t_{\min}s(X_{t_{\min}},t_{\min})
+
\sqrt{t_{\min}}\,Z.
$$

The score is evaluated at exactly $t_{\min}$, never below its training range.
Because the Stage 0 marginal is known, the experiment initializes from the
exact toy terminal law $\mathcal N(\mu,\tau^2+t_{\max})$.

Running the reverse integrator with the exact analytical score is a control for
time discretization and Monte Carlo error. Running the same integrator with the
learned score validates the complete learned generative pipeline. Paired runs
reuse the same initial samples and Brownian increments, so the remaining gap
between them diagnoses score approximation error.
