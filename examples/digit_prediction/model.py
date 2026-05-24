import mynn
import mytorch as mt


class Model(mynn.BaseModel):
    def __init__(self):

        self._l1 = mynn.LinearLayer(784, 64)
        self._l2 = mynn.LinearLayer(64, 10)

    def forward(self, batch: tuple[mt.Tensor, ...]) -> mt.Tensor:
        x = batch[0]
        y = self._l1(x)
        z = self._l2(y)
        return z

    def compute_loss(self, batch: tuple[mt.Tensor, ...], y_pred: mt.Tensor) -> mt.Tensor:
        y = batch[1]
        return y - y_pred

    def parameters(self) -> list[mt.Tensor]:
        return [*self._l1.parameters(), *self._l2.parameters()]
