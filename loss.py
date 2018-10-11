import numpy as np


class CrossEntropy:
    def forward(self, y_true, y_hat):
        self.y_hat = y_hat
        self.y_true = y_true
        self.loss = -np.sum(self.y_true * np.log(y_hat))
        return self.loss

    def backward(self):
        dz = -self.y_true / self.y_hat
        return dz
