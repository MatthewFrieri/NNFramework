import mytorch as mt
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, batch: tuple[mt.Tensor, ...]) -> mt.Tensor:
        pass

    @abstractmethod
    def compute_loss(self, batch: tuple[mt.Tensor, ...], y_pred: mt.Tensor) -> mt.Tensor:
        pass
