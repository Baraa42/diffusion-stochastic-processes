"""Minimal 1D denoising-score-matching experiment.

We will complete this file one mathematical component at a time.
"""

import torch


# Fixed data-distribution parameters, so the exact score is known.
MU = 2.0
TAU = 1.5

# Never sample t = 0: the DSM target contains 1 / sqrt(t).
T_MIN = 0.01
T_MAX = 2.0

BATCH_SIZE = 512


def sample_batch(batch_size: int):
    """Sample x_0, t, epsilon, x_t, and the conditional DSM target.

    Every returned tensor must have shape [batch_size, 1].
    """
    # TODO (Step 1): you will implement the five tensors here.
    raise NotImplementedError


if __name__ == "__main__":
    # We will add a shape-and-value sanity check after Step 1.
    pass
