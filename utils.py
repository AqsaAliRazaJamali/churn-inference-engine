import os
import random
import logging
import numpy as np

def setup_reproducibility(seed: int = 42) -> None:
    """Sets random seeds across libraries to ensure reproducible runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    logging.info(f"Reproducibility seed set to: {seed}")

def get_logger(name: str) -> logging.Logger:
    """Returns a standardized application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(name)