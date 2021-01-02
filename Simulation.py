# coding: utf-8
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def generate_test():
    x = np.array([[(np.random.rand() * 2 - 1) * np.math.pi]])
    # x = np.array([np.linspace(0, 2*np.math.pi, 100)])
    y = np.sin(x)
    for i in range(1000):
        x_temp = np.array([[(np.random.rand() * 2 - 1) * np.math.pi]])
        # x = np.array([np.linspace(0, 2*np.math.pi, 100)])
        y_temp = np.sin(x_temp)
        x = np.concatenate((x, x_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)
    return x, y

def lost_function(y, t):
    return 0.5 * np.sum((y-t)**2)/t.shape[0]

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据就是正确标签的内容，正确位置是1，其他位置为0
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx



class Convert:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None
    def forward(self, x):
        # 前向传播
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
    def backward(self, dout):
        # 反向传播
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class Loss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据
    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = lost_function(self.y, self.t)
        print self.loss
        return self.loss

    def backward(self, dout=1):
        dx = (self.y - self.t)
        return dx


class Network:

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] =np.load('SW1.npy') #weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] =np.load('Sb1.npy') #np.zeros(hidden_size1)
        self.params['W2'] =np.load('SW2.npy') #weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] =np.load('Sb2.npy') #np.zeros(hidden_size2)
        self.params['W3'] =np.load('SW3.npy') #weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] =np.load('Sb3.npy') #np.zeros(output_size)
        # 把层保存的有序的字典中
        self.layers = OrderedDict()
        self.layers['Convert1'] = Convert(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Convert2'] = Convert(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Convert3'] = Convert(self.params['W3'], self.params['b3'])
        self.lastLayer = Loss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t)
        # 反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 计算梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Convert1'].dW, self.layers['Convert1'].db
        grads['W2'], grads['b2'] = self.layers['Convert2'].dW, self.layers['Convert2'].db
        grads['W3'], grads['b3'] = self.layers['Convert3'].dW, self.layers['Convert3'].db
        return grads
    def generate_train(self):
        x = np.array([[(np.random.rand() * 2 - 1) * np.math.pi]])
        # x = np.array([np.linspace(0, 2*np.math.pi, 100)])
        y = np.sin(x)
        for i in range(1000):
            x_temp = np.array([[(np.random.rand() * 2 - 1) * np.math.pi]])
            # x = np.array([np.linspace(0, 2*np.math.pi, 100)])
            y_temp = np.sin(x_temp)
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)
        return x, y

    def startTraining(self, iter, learning_rate):
        x_train, y_train = self.generate_train()
        print x_train.shape
        print y_train.shape
        print x_train
        print y_train
        for i in range(iter):
            # 通过误差反向传播法求梯度
            grad = self.gradient(x_train, y_train)
            # 更新
            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                self.params[key] -= learning_rate * grad[key]
if __name__ == '__main__':
    netWork = Network(1, 1000, 100, 1)
    # for i in range(20):
    #     netWork.startTraining(100, 0.0001)
    # np.save("SW1", netWork.params['W1'])
    # np.save("Sb1", netWork.params['b1'])
    # np.save('SW2', netWork.params['W2'])
    # np.save('Sb2', netWork.params['b2'])
    # np.save('SW3', netWork.params['W3'])
    # np.save('Sb3', netWork.params['b3'])
#     训练结束，查看效果
    xt, yt = generate_test()

    y_p = netWork.predict(xt)
    accuracy =np.sum(y_p-yt)/yt.size
    print "accuracy:", accuracy
    ax = plt.gca()
    ax.set_title('data points')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.scatter(xt.T, y_p.T)
    plt.scatter(xt.T, yt.T)
    plt.show()



