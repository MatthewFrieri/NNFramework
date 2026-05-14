from __future__ import annotations
from typing import Optional
from mytorch.derivatives import *
import numpy as np


class Tensor:

    def __init__(self, data: np.ndarray | list):
        if isinstance(data, list):
            data = np.array(data)
        self._data = data
        self.shape = data.shape
        self.der: Optional[Derivative] = None
        self._reset_grad()

    @classmethod
    def rand(cls, shape: tuple):
        rng = np.random.default_rng()
        return cls(rng.random(shape))

    def _reset_grad(self):
        self.grad = np.zeros(self.shape)

    def back(self, on: float):
        if self.der is None:
            self.grad += on
            return
        for i, t in enumerate(self.der.operands):
            res = on * self.der.calc(i)
            t.back(res)

    def show_tree(self):
        print(self._get_tree(0, 2, {"": 96}))

    def _get_tree(self, indent: int, inc: int, name_map: dict):
        res = ""
        if self.der is None:
            key = id(self)
            if key not in name_map:
                name_map[key] = max(name_map.values()) + 1
            name = chr(name_map[key])
        else:
            name = self.der.__class__.__name__[3:]
        sym = ["o", ">"][indent // inc % 2]
        res += f"\n{' ' * indent}{sym} {name}"
        if self.der is None:
            return res
        for operand in self.der.operands:
            res += operand._get_tree(indent + inc, inc, name_map)
        return res

    def __add__(self, other: Tensor):
        y = Tensor(self._data + other._data)
        y.der = DerAdd(self, other)
        return y

    def __sub__(self, other: Tensor):
        y = Tensor(self._data - other._data)
        y.der = DerSub(self, other)
        return y

    def __mul__(self, other: Tensor):
        y = Tensor(self._data * other._data)
        y.der = DerMul(self, other)
        return y

    def __truediv__(self, other: Tensor):
        y = Tensor(self._data / other._data)
        y.der = DerDiv(self, other)
        return y
