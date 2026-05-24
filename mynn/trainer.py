from abc import abstractmethod

import mytorch as mt
from model import BaseModel
from data_loader import BaseDataLoader


class Trainer:

    def __init__(self, model: BaseModel):
        self._model = model

    def train(self, data_loader: BaseDataLoader):

        self.on_train_start()

        for epoch_idx in range(self._epochs):
            self.on_train_epoch_start(epoch_idx)

            for batch_idx, batch in enumerate(self.data_loader):
                self.on_train_batch_start(batch_idx, batch)

                y_pred = self._model.forward(batch)
                loss = self._model.compute_loss(batch, y_pred)
                loss.zero_grads()
                loss.back(on=1.0)

                self.on_train_batch_end(batch_idx, batch, y_pred, loss)

            self.on_train_epoch_end(epoch_idx)
        self.on_train_end()

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_train_epoch_start(self, epoch_idx: int):
        pass

    def on_train_epoch_end(self, epoch_idx: int):
        pass

    def on_train_batch_start(self, batch_idx: int, batch: tuple[mt.Tensor, ...]):
        pass

    def on_train_batch_end(
        self,
        batch_idx: int,
        batch: tuple[mt.Tensor, ...],
        y_pred: mt.Tensor,
        loss: mt.Tensor,
    ):
        pass

    def test(self, data_loader: BaseDataLoader):

        self.on_test_start()

        for batch_idx, batch in enumerate(self.data_loader):

            self.on_test_batch_start(batch_idx, batch)

            y_pred = self._model.forward(batch)
            loss = self._model.compute_loss(batch, y_pred)

            self.on_test_batch_end(batch_idx, batch, y_pred, loss)

        self.on_test_end()

    def on_test_start(self):
        pass

    def on_test_end(self):
        pass

    def on_test_batch_start(self, batch_idx: int, batch: tuple[mt.Tensor, ...]):
        pass

    def on_test_batch_end(
        self,
        batch_idx: int,
        batch: tuple[mt.Tensor, ...],
        y_pred: mt.Tensor,
        loss: mt.Tensor,
    ):
        pass
