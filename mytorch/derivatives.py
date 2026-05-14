import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytorch.tensor import Tensor


class Derivative:
    def __init__(self, *args):
        self.operands: list[Tensor] = list(args)

    def calc(self, wrt: int):
        raise NotImplementedError


class DerAdd(Derivative):
    def calc(self, wrt: int):
        return np.ones(self.operands[0].shape)


class DerSub(Derivative):
    def calc(self, wrt: int):
        if wrt == 0:
            return np.ones(self.operands[0].shape)
        if wrt == 1:
            return -np.ones(self.operands[0].shape)


class DerMul(Derivative):
    def calc(self, wrt: int):
        if wrt == 0:
            return self.operands[1]._data
        if wrt == 1:
            return self.operands[0]._data


class DerDiv(Derivative):
    def calc(self, wrt: int):
        if wrt == 0:
            return 1 / self.operands[1]._data
        if wrt == 1:
            return -self.operands[0]._data / (self.operands[1]._data ** 2)


class DerLog(Derivative):
    def calc(self, wrt: int):
        return 1 / self.operands[0]._data


class DerCos(Derivative):
    def calc(self, wrt: int):
        return -np.sin(self.operands[0]._data)
