import os
import random

import numpy as np
import torch

from src.common.config import CommonConfig, load_yaml, get_config_path


# Load YAML files
common_cfg = CommonConfig(**load_yaml(get_config_path("common.yaml"))["common"])


def set_global_seed(seed: int = common_cfg.random_seed) -> None:
    """ Set all relevant random seeds for reproducibility. """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """ Ensure deterministic behaviour for DataLoader workers. """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)