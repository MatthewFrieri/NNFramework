from typing import Iterable

import mytorch as mt
from mynn.optimizers.base_optimizer import BaseOptimizer


class AdamOptimizer(BaseOptimizer):

    def __init__(
        self,
        lr: float,
        beta_1: float,
        beta_2: float,
        epsilon: float,
    ):
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon

    def tune(self, params: Iterable[mt.Tensor]):
        pass  # TODO implement this
