import numpy as np


class Dense:
    def __init__(self, in_size, out_size, reg_lambda=0.0):
        self.W = np.random.normal(
            scale=1, size=(out_size, in_size)) * np.sqrt(2 / (in_size + out_size))
        self.b = np.zeros(out_size)
        self.reg_lambda = reg_lambda
        self.final_dW = 0
        self.final_db = 0

    def forward(self, x):
        self.x = x
        if np.array(x).shape[0] != self.W.shape[1]:
            print('X is not the same dimention as in_size')
        return (np.dot(self.W, self.x) + self.b)

    # честно верим, что эта функция работает правильно
    def get_reg_loss(self):
        return 0.5 * self.reg_lambda * (np.linalg.norm(self.W, ord='fro')**2)

    def backward(self,
                 dz,
                 learning_rate=0.001,
                 mini_batch=False,
                 update=True,
                 len_mini_batch=None):
        self.dW = np.outer(dz, self.x)
        self.db = dz
        self.dx = np.dot(dz, self.W)

        if (self.reg_lambda != 0):
            self.dW += self.reg_lambda * self.W

        if mini_batch == True:
            self.final_dW += self.dW
            self.final_db += self.db

        if update == True:
            if mini_batch == True:
                self.W = self.W - learning_rate * self.final_dW / len_mini_batch
                self.b = self.b - learning_rate * self.final_db / len_mini_batch
                self.final_dW = 0
                self.final_db = 0
            else:
                self.W = self.W - learning_rate * self.dW
                self.b = self.b - learning_rate * self.db

        return self.dx


class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x, train=True):
        if not train:
            self.mask = np.ones(*x.shape)
            return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dz, lr=0.001):
        return dz * self.mask