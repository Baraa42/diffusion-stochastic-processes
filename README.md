# Diffusion & Flow Matching for Stochastic Processes

This project implements score-based diffusion and flow matching from first
principles on analytically tractable distributions and stochastic processes.

Working principle:

> Known mathematical object → derive → implement → verify → increase complexity.

The first milestone is **1D Gaussian diffusion with analytical score
validation**. The experiment has not been implemented yet.

## Local development

```bash
poetry install
poetry run pytest
poetry run python experiments/00_gaussian_1d/train.py
poetry run jupyter lab
```
