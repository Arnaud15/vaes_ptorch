import enum
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils import gaussian_kl, gaussian_nll, mmd_rbf, sample_gaussian
