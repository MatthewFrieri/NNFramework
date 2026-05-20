import numpy as np

# ================== Abstract Classes ==================


class Derivative:
    def __init__(self):
        raise NotImplementedError

    def calc(self):
        raise NotImplementedError


class DerTS(Derivative):
    def __init__(self, t, s):
        self.t = t
        self.s = s
        self.shape = t.shape


class DerTT(Derivative):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.shape = t1.shape


class DerT(Derivative):
    def __init__(self, t):
        self.t = t
        self.shape = t.shape


# ================== Instantiable Classes ==================


class DerAddTS(DerTS):
    def calc(self):
        yield self.t, np.ones(self.shape)


class DerAddTT(DerTT):
    def calc(self):
        yield self.t1, np.ones(self.shape)
        yield self.t2, np.ones(self.shape)


class DerMulTS(DerTS):
    def calc(self):
        yield self.t, np.full(self.shape, self.s)


class DerMulTT(DerTT):
    def calc(self):
        yield self.t1, self.t2._data
        yield self.t2, self.t1._data


class DerPowTS(DerTS):
    def calc(self):
        yield self.t, self.s * self.t._data ** (self.s - 1)


class DerLog(DerT):
    def calc(self):
        yield self.t, 1 / self.t._data


class DerCos(DerT):
    def calc(self):
        yield self.t, -np.sin(self.t._data)
