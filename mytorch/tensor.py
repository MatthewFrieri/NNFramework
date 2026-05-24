from __future__ import annotations
from typing import Optional
from mytorch.const import Scalar
from mytorch.derivatives import *
import numpy as np


class Tensor:

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._data = data
        self._der: Optional[Derivative] = None
        self.shape: tuple = data.shape
        self.ndim = len(data.shape)
        self.zero_grads()

    @classmethod
    def _with_der(cls, data: np.ndarray, der: Derivative) -> Tensor:
        t = cls(data)
        t._der = der
        return t

    @classmethod
    def rand(cls, shape: tuple) -> Tensor:
        rng = np.random.default_rng()
        return cls(rng.random(shape))

    def zero_grads(self):
        self.grade = np.zeros(self.shape)
        if self._der is not None:
            for t, _ in self._der.calc():
                t.zero_grads()

    def back(self, on: float):
        if self._der is None:
            self.grad += on
            return
        for t, res in self._der.calc():
            t.back(on * res)

    def computation_tree(self) -> str:
        return self._get_tree(0, 2, {"": 96})

    def _get_tree(self, indent: int, inc: int, name_map: dict) -> str:
        res = ""
        if self._der is None:
            key = id(self)
            if key not in name_map:
                name_map[key] = max(name_map.values()) + 1
            name = chr(name_map[key])
        else:
            name = self._der.__class__.__name__[3:]
        res += f"\n{' ' * indent}> {name}"
        if self._der is None:
            return res
        if isinstance(self._der, DerTS):
            res += self._der.t._get_tree(indent + inc, inc, name_map)
            res += f"\n{' ' * (indent + inc)}> {self._der.s}"
        if isinstance(self._der, DerTT):
            res += self._der.t1._get_tree(indent + inc, inc, name_map)
            res += self._der.t2._get_tree(indent + inc, inc, name_map)
        if isinstance(self._der, DerT):
            res += self._der.t._get_tree(indent + inc, inc, name_map)
        return res

    def __add__(self, other) -> Tensor:
        return Tensor.add(self, other)

    def __radd__(self, other) -> Tensor:
        return Tensor.add(other, self)

    def __sub__(self, other) -> Tensor:
        return Tensor.sub(self, other)

    def __rsub__(self, other) -> Tensor:
        return Tensor.sub(other, self)

    def __mul__(self, other) -> Tensor:
        return Tensor.mul(self, other)

    def __rmul__(self, other) -> Tensor:
        return Tensor.mul(other, self)

    def __truediv__(self, other) -> Tensor:
        return Tensor.div(self, other)

    def __rtruediv__(self, other) -> Tensor:
        return Tensor.div(other, self)

    @classmethod
    def mat_mul(cls, t1, t2) -> Tensor:
        pass

    @classmethod
    def add(cls, x1, x2) -> Tensor:
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            z = cls._with_der(x2._data + x1, DerAddTS(x2, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls._with_der(x1._data + x2, DerAddTS(x1, x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            z = cls._with_der(x1._data + x2._data, DerAddTT(x1, x2))
        return z

    @classmethod
    def sub(cls, x1, x2) -> Tensor:
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            y = cls.mul(x2, -1)
            z = cls._with_der(y._data + x1, DerAddTS(y, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls._with_der(x1._data - x2, DerAddTS(x1, -x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            y = cls.mul(x2, -1)
            z = cls._with_der(x1._data + y._data, DerAddTT(x1, y))
        return z

    @classmethod
    def mul(cls, x1, x2) -> Tensor:
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            z = cls._with_der(x2._data * x1, DerMulTS(x2, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls._with_der(x1._data * x2, DerMulTS(x1, x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            z = cls._with_der(x1._data * x2._data, DerMulTT(x1, x2))
        return z

    @classmethod
    def div(cls, x1, x2) -> Tensor:
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            y = cls.pow(x2, -1)
            z = cls._with_der(y._data * x1, DerMulTS(y, x1))
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls._with_der(x1._data / x2, DerMulTS(x1, 1 / x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            y = cls.pow(x2, -1)
            z = cls._with_der(x1._data * y._data, DerMulTT(x1, y))
        return z

    @classmethod
    def pow(cls, x1, x2) -> Tensor:
        if isinstance(x1, Scalar) and isinstance(x2, Tensor):
            raise NotImplementedError
        elif isinstance(x1, Tensor) and isinstance(x2, Scalar):
            z = cls._with_der(x1._data**x2, DerPowTS(x1, x2))
        elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
            raise NotImplementedError
        return z

    @classmethod
    def log(cls, t: Tensor) -> Tensor:
        return cls._with_der(np.log(t._data), DerLog(t))

    @classmethod
    def cos(cls, t: Tensor) -> Tensor:
        return cls._with_der(np.cos(t._data), DerCos(t))
