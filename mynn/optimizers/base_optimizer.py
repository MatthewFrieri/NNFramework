from abc import ABC, abstractmethod
from typing import Iterable

import mytorch as mt


class BaseOptimizer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def tune(self, params: Iterable[mt.Tensor]):
        pass
