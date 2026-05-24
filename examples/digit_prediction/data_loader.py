import mynn
import mytorch as mt
import numpy as np
import struct
from array import array


class DataLoader(mynn.BaseDataLoader):

    def __init__(
        self,
        batch_size: int,
        train_images_path: str,
        train_labels_path: str,
        test_images_path: str,
        test_labels_path: str,
    ):
        self._batch_size = batch_size
        self._train_x, self._train_y = self._read_images_labels(train_images_path, train_labels_path)
        self._test_x, self._test_y = self._read_images_labels(test_images_path, test_labels_path)

    def _read_images_labels(self, images_path: str, labels_path: str):
        labels = []
        with open(labels_path, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError("Magic number mismatch, expected 2049, got {}".format(magic))
            labels = array("B", file.read())

        with open(images_path, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch, expected 2051, got {}".format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            images[i][:] = img

        return images, labels

    def train_data(self):
        for start in range(0, len(self._train_x), self._batch_size):
            end = start + self._batch_size
            yield mt.Tensor(self._train_x[start:end]), mt.Tensor(self._train_y[start:end])

    def test_data(self):
        for start in range(0, len(self._test_x), self._batch_size):
            end = start + self._batch_size
            yield mt.Tensor(self._test_x[start:end]), mt.Tensor(self._test_y[start:end])
