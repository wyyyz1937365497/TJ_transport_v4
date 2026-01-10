"""随机种子设置 - 支持所有常用库"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    设置所有随机数生成器的种子

    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式（会影响性能）
    """
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确定性模式（可选，会降低性能）
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Python hash seed
    if hasattr(os, '.environ'):
        os.environ['PYTHONHASHSEED'] = str(seed)


import os  # type: ignore
