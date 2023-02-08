# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960


import numpy as np
import matplotlib.pyplot as plt


# Endorsed by instructor. Used for reading .gz files.
def read_idx(filename):
    import gzip
    import struct
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# step activation function u(·)
def step(x):
    return np.heaviside(x, 1)  # if x<0 output 0


if __name__ == '__main__':
    train_inputs = read_idx("train-images-idx3-ubyte.gz")
    train_labels = read_idx("train-labels-idx1-ubyte.gz")
    test_input = read_idx("t10k-images-idx3-ubyte.gz")
    test_labels = read_idx("t10k-labels-idx1-ubyte.gz")

    # (d) Build and train the network.
    # 0)Given η, ε, n:
    lr = 1
    lr_set = [1, 0.1, 0.1, 0.1]
    lr_flag = 0
    eps = 0
    # n = 50
    N = [50, 1000, 60000, 60000, 60000, 60000]
    # 1) Initialize W ∈ R10×784 randomly.
    W_init = np.random.uniform(-1, 1, (10, 784))

    for n in N:
        if n == 60000:
            lr = lr_set[lr_flag]
            if lr_flag > 0:
                eps = 0.113
                W_init = np.random.uniform(-1, 1, (10, 784))
            lr_flag += 1
        # 2) Initialize epoch = 0.
        epoch = 0
        # 3) Initialize errors(epoch) = 0, for epoch = 0, 1, ....
        array_errors = []
        W = W_init
        flag = True
        print('------Training------')
        while flag:
            num_errors = 0
            # 3.1.1 Count the misclassification errors.
            for i in range(n):
                x_i = train_inputs[i].reshape((784, 1))
                # 3.1.1.1 Calculate the induced local fields with the current training sample and weights.
                v = W @ x_i
                # print(v)
                # 3.1.1.2 Choose the output neuron with the largest induced local field.
                v_j = np.argmax(v)
                # 3.1.1.3 If j is not the same as the input label, then errors(epoch) ← errors(epoch) + 1.
                if v_j != train_labels[i]:
                    num_errors += 1

                # 3.1.3 update the weights. For convenience, this is combined to the for loop in 3.1.1
                d_xi = np.zeros((10, 1))
                d_xi[train_labels[i]] = 1
                W = W + lr * (d_xi - step(v)) @ x_i.T

            # print('Epoch: {}, errors: {}, ε: {}'.format(epoch, num_errors, eps))
            # 3.1.2 epoch ← epoch + 1.
            epoch += 1
            array_errors.append(num_errors)

            if num_errors / n <= eps or epoch >= 500:
                flag = False

        print('n : {}, lr: {}, ε : {}, epochs: {}, errors: {}'.format(n, lr, eps, epoch, num_errors))
        # (f) Plot the epoch number vs.the number of misclassification errors
        array_epoch = list(range(epoch))
        plt.figure()
        plt.title('Epochs vs. Misclassifications n = {}, η = {}, ε = {}'.format(n, lr, eps))
        plt.xlabel('Epochs')
        plt.ylabel('Misclassifications')
        plt.plot(array_epoch, array_errors)
        plt.show()

        # (e) Test the network with test set.
        num_errors_test = 0
        for i in range(10000):
            x_i = test_input[i].reshape((784, 1))
            # 2.1 Calculate the induced local fields with the current training sample and weights.
            v = W @ x_i
            # print(v)
            # 2.2 Choose the output neuron with the largest induced local field.
            v_j = np.argmax(v)
            # 2.3 If j is not the same as the input label, then errors ← errors + 1.
            if v_j != test_labels[i]:
                num_errors_test += 1
        # (f) Record the percentage of misclassified test samples
        print('------Testing------')
        print('errors : {}, error rate: {}'.format(num_errors_test, num_errors_test / 10000))
