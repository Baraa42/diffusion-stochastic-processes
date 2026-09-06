# Diffusion & Flow Matching for Stochastic Processes

This project implements score-based diffusion and flow matching from first
principles on analytically tractable distributions and stochastic processes.

> Known mathematical object → derive → implement → verify → increase complexity.

**Stage 0 — 1D Gaussian diffusion is complete.** It includes DSM score
learning, analytical score validation, and reverse Euler-Maruyama sampling.

Detailed derivations and results:

- [Stage 0 experiment](experiments/00_gaussian_1d/README.md)
- [Gaussian diffusion](notes/00_gaussian_diffusion.md)
- [Denoising score matching](notes/01_score_matching.md)
- [Reverse SDE](notes/02_reverse_sde.md)

Stage 1 and flow matching have not been implemented.

## Local development

~~~bash
poetry install
poetry run pytest
poetry run python experiments/00_gaussian_1d/train.py
poetry run jupyter lab
~~~
