from layers import Dense, Dropout
from loss import CrossEntropy
from activations import ReLu, Softmax
import numpy as np


class MnistNetMiniBatch:
    def __init__(self):
        self.d1_layer = Dense(784, 100)
        self.a1_layer = ReLu()
        self.drop1_layer = Dropout(0.5)

        self.d2_layer = Dense(100, 50)
        self.a2_layer = ReLu()
        self.drop2_layer = Dropout(0.25)

        self.d3_layer = Dense(50, 10)
        self.a3_layer = Softmax()

    def forward(self, x, train=True):
        net = self.d1_layer.forward(x)
        net = self.a1_layer.forward(net)
        net = self.drop1_layer.forward(net, train)

        net = self.d2_layer.forward(net)
        net = self.a2_layer.forward(net)
        net = self.drop2_layer.forward(net, train)

        net = self.d3_layer.forward(net)
        net = self.a3_layer.forward(net)

        return (net)

    def backward(self,
                 dz,
                 learning_rate=0.01,
                 mini_batch=True,
                 update=False,
                 len_mini_batch=None):

        dz = self.a3_layer.backward(dz)
        dz = self.d3_layer.backward(
            dz,
            learning_rate=learning_rate,
            mini_batch=mini_batch,
            update=update,
            len_mini_batch=len_mini_batch)

        dz = self.drop2_layer.backward(dz)
        dz = self.a2_layer.backward(dz)
        dz = self.d2_layer.backward(
            dz,
            learning_rate=learning_rate,
            mini_batch=mini_batch,
            update=update,
            len_mini_batch=len_mini_batch)

        dz = self.drop1_layer.backward(dz)
        dz = self.a1_layer.backward(dz)
        dz = self.d1_layer.backward(
            dz,
            learning_rate=learning_rate,
            mini_batch=mini_batch,
            update=update,
            len_mini_batch=len_mini_batch)

        return dz


def compute_acc(X_test, Y_test, net):
    '''Not one-hot encoded format'''
    acc = 0.0
    for i in range(X_test.shape[0]):
        y_h = net.forward(X_test[i])
        y = np.argmax(y_h)
        if (y == Y_test[i]):
            acc += 1.0
    return acc / Y_test.shape[0]


if __name__ == 'main':
    loss = CrossEntropy()
    net = MnistNetMiniBatch()
    learning_rate = 0.001
    L_train = []
    L_test = []
    Acc_train = []
    Acc_test = []
    len_mini_batch = 10
    for it in range(100):
        L_acc = 0.
        sh = list(range(train_x.shape[0]))
        np.random.shuffle(sh)
        for i in range(train_x.shape[0]):
            x = train_x[sh[i]]
            y = train_y_oh[sh[i]]
            y_h = net.forward(x)
            L = loss.forward(y, y_h)
            L_acc += L
            dz = loss.backward()
            if i % len_mini_batch == 0:
                dz = net.backward(
                    dz,
                    learning_rate,
                    update=True,
                    len_mini_batch=len_mini_batch)
            else:
                dz = net.backward(dz, learning_rate)
        L_acc /= train_y_oh.shape[0]
        L_train.append(L_acc)
        acc = compute_acc(train_x, train_y, net)
        Acc_train.append(acc)
        L_e_acc = 0.
        for i in range(test_x.shape[0]):
            x = test_x[i]
            y = test_y_oh[i]
            y_h = net.forward(x)
            L = loss.forward(y, y_h)
            L_e_acc += L
        L_e_acc /= test_y_oh.shape[0]
        L_test.append(L_e_acc)
        acc = compute_acc(test_x, test_y, net)
        Acc_test.append(acc)

        learning_rate = learning_rate * 0.99

        print(
            "{} epoch. Train : {} . Test : {}. acc : {} . val_acc: {}".format(
                it + 1, L_acc, L_e_acc, Acc_train[-1], Acc_test[-1]))
