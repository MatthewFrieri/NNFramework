from typing import Iterable
import numpy as np

import mytorch as mt
from mynn.optimizers.base_optimizer import BaseOptimizer


class AdamOptimizer(BaseOptimizer):

    def __init__(
        self,
        lr: float,
        beta_1: float,
        beta_2: float,
        epsilon: float,
        maximize: bool = False,
        amsgrad: bool = False,
    ):
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._maximize = maximize
        self._amsgrad = amsgrad
        self._t = 0
        self._m_old: dict[int, np.ndarray] = {}
        self._v_old: dict[int, np.ndarray] = {}
        self._v_max_old: dict[int, np.ndarray] = {}

    def tune(self, params: Iterable[mt.Tensor]):

        self._t += 1
        for i, param in enumerate(params):
            g = param.grad

            if all(g == 0.0):
                continue

            if i not in self._m_old:
                self._m_old[i] = np.zeros_like(g)
                self._v_old[i] = np.zeros_like(g)
                self._v_max_old[i] = np.zeros_like(g)

            if self._maximize:
                g = -g

            m = self._beta_1 * self._m_old[i] + (1 - self._beta_1) * g
            v = self._beta_2 * self._v_old[i] + (1 - self._beta_2) * g**2

            m_hat = m / (1 - self._beta_1**self._t)

            if self._amsgrad:
                v_max = np.maximum(self._v_max_old, self._v)
                v_hat = v_max / (1 - self._beta_2**self._t)
            else:
                v_hat = v / (1 - self._beta_2**self._t)

            param._data -= self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
