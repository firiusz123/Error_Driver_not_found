import os
import random

import numpy as np
import torch


LABEL_TO_ID = {"alert": 0, "drowsy": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
