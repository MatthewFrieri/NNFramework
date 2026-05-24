from typing import Iterable

import mytorch as mt
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, batch: tuple[mt.Tensor, ...]) -> mt.Tensor:
        """
        A forward pass through the model.
        Returns a Tensor of predictions with shape (batch_size, out_dim).
        """
        pass

    @abstractmethod
    def compute_loss(self, batch: tuple[mt.Tensor, ...], y_pred: mt.Tensor) -> mt.Tensor:
        """
        Computes the loss for a batch.
        Returns a Tensor of a scalar loss.
        """
        pass

    @abstractmethod
    def parameters(self) -> Iterable[mt.Tensor]:
        pass
