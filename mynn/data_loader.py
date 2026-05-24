from abc import ABC, abstractmethod


class BaseDataLoader(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train_data(self):
        """
        Yields a batch of training data.
        A batch is a tuple of Tensors, where each tensor has shape (batch_size, ?).
        """
        pass

    @abstractmethod
    def test_data(self):
        """
        Yields a batch of testing data.
        A batch is a tuple of Tensors, where each tensor has shape (batch_size, ?).
        """
        pass
