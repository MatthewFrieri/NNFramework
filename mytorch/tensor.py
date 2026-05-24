from __future__ import annotations
from typing import Optional
from mytorch.const import Scalar
from mytorch.derivatives import *
import numpy as np


class Tensor:

    def __init__(self, data: np.ndarray | list):
        if isinstance(data, list):
            data = np.array(data)
        self._data = data
        self.shape: tuple = data.shape
        self.ndim = len(data.shape)
        self.der: Optional[Derivative] = None
        self.zero_grads()

    @classmethod
    def with_der(cls, data: np.ndarray | list, der: Derivative):
        t = cls(data)
        t.der = der
        return t

    @classmethod
    def rand(cls, shape: tuple):
        rng = np.random.default_rng()
        return cls(rng.random(shape))

    def zero_grads(self):
        self.grade = np.zeros(self.shape)
        if self.der is not None:
            for t, _ in self.der.calc():
                t.zero_grads()

    def back(self, on: float):
        if self.der is None:
            self.grad += on
            return
        for t, res in self.der.calc():
            t.back(on * res)

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
        res += f"\n{' ' * indent}> {name}"
        if self.der is None:
            return res
        if isinstance(self.der, DerTS):
            res += self.der.t._get_tree(indent + inc, inc, name_map)
            res += f"\n{' ' * (indent + inc)}> {self.der.s}"
        if isinstance(self.der, DerTT):
            res += self.der.t1._get_tree(indent + inc, inc, name_map)
            res += self.der.t2._get_tree(indent + inc, inc, name_map)
        if isinstance(self.der, DerT):
            res += self.der.t._get_tree(indent + inc, inc, name_map)
        return res

    def __add__(self, other):
        return Tensor.add(self, other)

    def __radd__(self, other):
        return Tensor.add(other, self)

    def __sub__(self, other):
        return Tensor.sub(self, other)

    def __rsub__(self, other):
        return Tensor.sub(other, self)

    def __mul__(self, other):
        return Tensor.mul(self, other)

    def __rmul__(self, other):
        return Tensor.mul(other, self)

    def __truediv__(self, other):
        return Tensor.div(self, other)

    def __rtruediv__(self, other):
        return Tensor.div(other, self)

    @classmethod
    def mat_mul(cls, t1, t2):
        pass

    @classmethod
    def add(cls, x1, x2):
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            z = cls.with_der(x2._data + x1, DerAddTS(x2, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls.with_der(x1._data + x2, DerAddTS(x1, x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            z = cls.with_der(x1._data + x2._data, DerAddTT(x1, x2))
        return z

    @classmethod
    def sub(cls, x1, x2):
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            y = cls.mul(x2, -1)
            z = cls.with_der(y._data + x1, DerAddTS(y, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls.with_der(x1._data - x2, DerAddTS(x1, -x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            y = cls.mul(x2, -1)
            z = cls.with_der(x1._data + y._data, DerAddTT(x1, y))
        return z

    @classmethod
    def mul(cls, x1, x2):
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            z = cls.with_der(x2._data * x1, DerMulTS(x2, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls.with_der(x1._data * x2, DerMulTS(x1, x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            z = cls.with_der(x1._data * x2._data, DerMulTT(x1, x2))
        return z

    @classmethod
    def div(cls, x1, x2):
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            y = cls.pow(x2, -1)
            z = cls.with_der(y._data * x1, DerMulTS(y, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls.with_der(x1._data / x2, DerMulTS(x1, 1 / x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            y = cls.pow(x2, -1)
            z = cls.with_der(x1._data * y._data, DerMulTT(x1, y))
        return z

    @classmethod
    def pow(cls, x1, x2):
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            raise NotImplementedError
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls.with_der(x1._data**x2, DerPowTS(x1, x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            raise NotImplementedError
        return z

    @classmethod
    def log(cls, t: Tensor):
        return cls.with_der(np.log(t._data), DerLog(t))

    @classmethod
    def cos(cls, t: Tensor):
        return cls.with_der(np.cos(t._data), DerCos(t))
