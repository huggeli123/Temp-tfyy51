import yaml
import torch
import numpy as np
import random

# ============================================================
# Configuration
# ============================================================

def load_config(config_path="config.yaml"):
    """
    Load settings from a YAML configuration file.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)