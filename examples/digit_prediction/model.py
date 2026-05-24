import mynn
import mytorch as mt


class Model(mynn.BaseModel):
    def __init__(self):

        self.l1 = mynn.LinearLayer(256, 64)
        self.l2 = mynn.LinearLayer(64, 10)

    def forward(self, batch: mt.Tensor):
        y = self.l1(batch)
        z = self.l2(y)
        return z
