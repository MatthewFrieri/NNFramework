from abc import ABC, abstractmethod


class BaseDataLoader(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train_data(self):
        pass

    @abstractmethod
    def test_data(self):
        pass
