# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960


import numpy as np
import matplotlib.pyplot as plt


def activation_1(v):
    return np.tanh(v)


def activation_1_prime(v):
    return 1 - pow(np.tanh(v), 2)


def activation_2(v):
    return v


seed = 655490960
np.random.seed(seed)
n = 300
# 1. Draw n=300 real numbers uniformly at random on [0, 1], call them x1,...,xn.
X = np.random.uniform(0, 1, (n, 1))
# 2. Draw n real numbers uniformly at random on [− 1 , 1 ], call them ν1,...,νn.
V = np.random.uniform(-1 / 10, 1 / 10, (n, 1))
# 3. Let di = sin(20xi) + 3xi + νi, i = 1, ..., n.
D = np.sin(20 * X) + 3 * X + V

# Plot the points(xi, di), i = 1, ..., n.
plt.figure()
plt.title('X vs. D n = {}'.format(n))
plt.xlabel('X')
plt.ylabel('D')
plt.scatter(X, D)
plt.savefig('X_vs_Label.pdf')
plt.show()

lr = 0.001
eps = 0.005

if __name__ == '__main__':
    N = 24
    # Let w denote the vector of all these 3N + 1 weights.
    W = np.random.uniform(-0.1, 0.1, 3 * N + 1)
    # weights before first layer
    w1_init = W[0:N]
    # weights before second layer
    w2_init = W[2 * N:3 * N]
    # bias for first layer
    b1_init = W[N:2 * N]
    # bias for second layer
    b2_init = W[3 * N]

    epoch = 0
    error_array = []

    w1 = w1_init
    w2 = w2_init
    b1 = b1_init
    b2 = b2_init

    flag = True
    print('------Training------')
    while flag:
        error = 0
        # online training
        for i in range(n):
            # feed forward
            v1 = w1 * X[i] + b1
            y1 = activation_1(v1)
            v2 = w2.transpose() * y1
            y2 = activation_2(np.sum(v2) + b2)
            # calculate errors
            error += pow(D[i] - y2, 2)
            # back forward
            b2 = b2 + lr * 1 * (D[i] - y2) * 1
            w2 = w2 + lr * y1 * (D[i] - y2) * 1
            b1 = b1 + lr * 1 * w2 * (D[i] - y2) * 1 * activation_1_prime(v1)
            w1 = w1 + lr * X[i] * w2 * (D[i] - y2) * 1 * activation_1_prime(v1)

        MSE = error / n
        error_array.append(MSE)
        # modify the gradient descent algorithm by decreasing η
        if epoch > 1 and (error_array[-1] > error_array[-2]):
            lr = lr * 0.9
        epoch += 1
        if epoch > 1000000 or MSE < eps:
            flag = False

    # Plot the number of epochs vs the MSE in the bp algorithm.
    epoch_array = range(epoch)
    plt.figure()
    plt.title('epoch vs. MSE η = {}'.format(lr))
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(epoch_array, error_array)
    plt.savefig('Epoch_vs_MSE.pdf')
    plt.show()

    outputs = []
    for i in range(n):
        v1 = w1 * X[i] + b1
        y1 = activation_1(v1)
        v2 = w2.transpose() * y1
        y2 = activation_2(np.sum(v2) + b2)
        outputs.append(y2)

    # Plot the curve f(x, w0)
    plt.figure()
    plt.title('curve (x,f(x,w0)) vs. Label')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(X, D, label='Label')
    plt.scatter(X, outputs, label='Output')
    plt.savefig('Output_vs_Label.pdf')
    plt.show()
