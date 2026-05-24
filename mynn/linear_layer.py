import mytorch as mt


class LinearLayer:

    def __init__(self, in_dim: int, out_dim: int):
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._weights = mt.Tensor.rand((in_dim, out_dim))
        self._biases = mt.Tensor.rand(out_dim)

    def __call__(self, batch: mt.Tensor) -> mt.Tensor:
        y = mt.Tensor.mat_mul(batch, self._weights)
        z = y + self._biases
        return z

    def parameters(self) -> list[mt.Tensor]:
        return [self._weights, self._biases]
