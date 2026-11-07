"""Seed control for reproducible runs.

Libraries that take an explicit seed (LightGBM, Optuna) receive it directly from the
caller; this module only seeds the global generators so that nothing else draws from an
unseeded source.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` and NumPy's legacy global generator, and log the seed."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Global seed set to %d", seed)
