import numpy as np


class Softmax:
    def forward(self, x):
        x = x - np.max(x)
        self.p = np.exp(x) / np.sum(np.exp(x))
        return self.p

    def backward(self, dz):
        jacobian = np.diag(dz)
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                if i == j:
                    jacobian[i][j] = self.p[i] * (1 - self.p[j])
                else:
                    jacobian[i][j] = -self.p[i] * self.p[j]
        return np.matmul(dz, jacobian)


class ReLu:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dz):
        dz[self.x < 0] = 0
        return dz