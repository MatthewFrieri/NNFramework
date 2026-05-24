import os
import mynn
from .data_loader import DataLoader
from .model import Model

BATCH_SIZE = 100
EPOCHS = 2

data_loader = DataLoader(
    BATCH_SIZE,
    os.path.join(os.path.dirname(__file__), "data", "train-images.idx3-ubyte"),
    os.path.join(os.path.dirname(__file__), "data", "train-labels.idx1-ubyte"),
    os.path.join(os.path.dirname(__file__), "data", "test-images.idx3-ubyte"),
    os.path.join(os.path.dirname(__file__), "data", "test-labels.idx1-ubyte"),
)

model = Model()

trainer = mynn.Trainer(model, data_loader, EPOCHS)
trainer.train()
