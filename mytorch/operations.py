from mytorch.derivatives import *
from mytorch.tensor import Tensor
import numpy as np


def log(t: Tensor):
    y = Tensor(np.log(t._data))
    y.der = DerLog(t)
    return y


def cos(t: Tensor):
    y = Tensor(np.cos(t._data))
    y.der = DerCos(t)
    return y
