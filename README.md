# Diffusion and Stochastic Processes

## Block C: Minimal PyTorch DSM

A deliberately small experiment that checks denoising score matching against a
one-dimensional Gaussian score that we know exactly.

## Mathematical setup

We fix

\[
X_0 \sim \mathcal N(\mu, \tau^2)
\]

and use the forward perturbation

\[
X_t = X_0 + \sqrt{t}\,\varepsilon,
\qquad \varepsilon \sim \mathcal N(0,1).
\]

The noisy conditional DSM target is

\[
-\frac{\varepsilon}{\sqrt t},
\]

while the exact marginal score used for validation is

\[
s^*(x,t) = -\frac{x-\mu}{\tau^2+t}.
\]

## Initial choices

- `mu = 2.0`
- `tau = 1.5`
- `t ~ Uniform(0.01, 2.0)`
- one tiny MLP, implemented directly in PyTorch
- no diffusion libraries and no reverse sampling

We exclude `t = 0` because the DSM target contains `1 / sqrt(t)` and its
variance grows as `1 / t` near zero.

## Local development

Install the locked environment:

```bash
poetry install
```

Run the Python experiment:

```bash
poetry run python experiment.py
```

Start JupyterLab:

```bash
poetry run jupyter lab
```

In a notebook, select the kernel whose Python executable belongs to this
project's Poetry virtual environment.

## Structure

- `experiment.py`: sampling, tiny network, training, and validation
- `results/`: generated plots (not committed)
- `pyproject.toml`: Poetry environment and dependencies

The experiment is developed in small stages. Important mathematical code is
left for the learner to complete rather than supplied all at once.
