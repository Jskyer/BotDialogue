import numpy as np

# 二分类激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 多分类激活函数
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# loss 交叉熵
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    h = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + h)) / batch_size

# w: 矩阵x 展平后的一维表示
def numeric_grad(f, x):
    h = 1e-4
    w = x.flatten()
    grads = np.zeros_like(w)
    print(w.size)

    for idx in range(w.size):
        tmp = w[idx]
        w[idx] = tmp + h
        h1 = f(w)
        w[idx] = tmp - h
        h2 = f(w)

        grads[idx] = (h1 - h2) / (2 * h)
        w[idx] = tmp

    return grads


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = softmax(a2)
        return z2

    # x 输入数据，t 监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        f = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numeric_grad(f, self.params['W1'])
        grads['b1'] = numeric_grad(f, self.params['b1'])
        grads['W2'] = numeric_grad(f, self.params['W2'])
        grads['b2'] = numeric_grad(f, self.params['b2'])
        return grads


net = TwoLayerNet(input_size=2, hidden_size=100, output_size=10)
print(net.params['W1'])
print(net.params['b1'])
print(net.params['W2'])
print(net.params['b2'])

# x = np.random.rand(100, 2)
# t = np.random.rand(100, 10)
# grads = net.numerical_gradient(x, t)
#
# print(grads['W1'].shape)
# print(grads['W1'])
# print("=============")
# print(grads['b1'].shape)
# print(grads['b1'])
# print("=============")

# print(grads['W2'].shape)
# print(grads['b2'].shape)


