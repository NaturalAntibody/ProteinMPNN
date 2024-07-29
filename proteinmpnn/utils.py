from typing import Optional, TextIO
import numpy as np
import torch


import random


def set_random_seed(seed: Optional[int] = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_line_count(file_handle: TextIO) -> int:
    line_count = sum(1 for _ in file_handle)
    file_handle.seek(0, 0)
    return line_count