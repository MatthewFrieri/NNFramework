from typing import Optional

import mytorch as mt
from mynn.model import BaseModel
from mynn.data_loader import BaseDataLoader
from mynn.optimizers.base_optimizer import BaseOptimizer
from mynn.optimizers.adam_optimizer import AdamOptimizer


class Trainer:

    def __init__(
        self,
        model: BaseModel,
        data_loader: BaseDataLoader,
        optimizer: Optional[BaseOptimizer],
        epochs: int,
    ):
        self._model = model
        self._model_params = model.parameters()
        self._data_loader = data_loader
        if optimizer is None:
            optimizer = AdamOptimizer(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self._optimizer = optimizer
        self._epochs = epochs

    def train(self):
        """
        Train the model.
        Iterates through epochs and batches, calling forward, computing loss, then backproping on that loss.
        Uses the optimizer to tune model weights at each step.
        """
        self.on_train_start()

        for epoch_idx in range(self._epochs):
            self.on_train_epoch_start(epoch_idx)

            for batch_idx, batch in enumerate(self._data_loader.train_data()):
                self.on_train_batch_start(batch_idx, batch)

                y_pred = self._model.forward(batch)
                loss = self._model.compute_loss(batch, y_pred)
                loss.zero_grads()
                loss.back(on=1.0)
                self._optimizer.tune(self._model_params)

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

    def test(self):
        """
        Test the model.
        Iterates through epochs and batches, calling forward and computing loss.
        """

        self.on_test_start()

        for batch_idx, batch in enumerate(self._data_loader.test_data()):

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
