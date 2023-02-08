# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960

import numpy as np
import matplotlib.pyplot as plt


# step activation function u(·)
def step(x):
    if x >= 0:
        return 1
    else:
        return 0


font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

# (a) Pick (your code should pick it) w0 uniformly at random on [−1/4,1/4].
# (b) Pick w1 uniformly at random on [−1, 1].
# (c) Pick w2 uniformly at random on [−1, 1].
w0 = np.random.uniform(-0.25, 0.25, 1)
w1 = np.random.uniform(-1, 1, 1)
w2 = np.random.uniform(-1, 1, 1)

# (d) Pick n = 100 vectors x1,...,xn independently and uniformly at
# random on [−1, 1]^2, call the collection of these vectors S.
# (e) Let S1 ⊂ S denote the collection of all x = [x1 x2] ∈ S
# satisfying [1 x1 x2][w0 w1, w2]^T ≥ 0.
# (f) Let S0 ⊂ S denote the collection of all x = [x1 x2] ∈ S
# satisfying [1 x1 x2][w0 w1, w2]^T < 0.
S, S1, S0 = [], [], []
x1S1, x2S1 = [], []
x1S0, x2S0 = [], []
x1line, x2line = [], []
Label = []

# (n) Do the same experiments with n = 1000 samples.
# N = 100
# N = 1000
Sizes = [100, 1000]

for N in Sizes:
    print('------N = {}------'.format(N))
    for i in range(N):
        x_i = np.random.uniform(-1, 1, 2)
        S.append(x_i)

        output = step(w0 + w1 * x_i[0] + w2 * x_i[1])
        Label.append(output)

        x1_val = x_i[0]
        x1line.append(x1_val)
        x2_val = (-w0 - w1 * x_i[0]) / w2
        x2line.append(x2_val)

        if output == 1:
            S1.append(x_i)
            x1S1.append(x_i[0])
            x2S1.append(x_i[1])
        elif output == 0:
            S0.append(x_i)
            x1S0.append(x_i[0])
            x2S0.append(x_i[1])

    # (g) In one plot, show the line w0 + w1x1 + w2x2 = 0, with x1 being the
    # “x-axis” and x2 being the “y-axis.”
    # In the same plot, show all the points in S1 and all the points in S0.
    # Use different symbols for S0 and S1.
    plt.figure()
    plt.title("Perceptron Classification Interpolation", fontdict=font1)
    plt.xlabel("x1", fontdict=font2)
    plt.ylabel("x2", fontdict=font2)
    plt.scatter(x1S0, x2S0, marker='x')
    plt.scatter(x1S1, x2S1, marker='.')
    plt.plot(x1line, x2line, 'k-')
    plt.legend(["x in S0", "x in S1", "line"], loc="upper left")
    plt.show()
    # plt.draw()
    print('Optimal Weights: w0={}, w1={}, w2={}'.format(w0, w1, w2))

    # (h) Use the perceptron training algorithm to find the weights that can
    # separate the two classes S0 and S1
    # (j) Repeat the same experiment with η = 10. Do not change w0, w1, w2,
    # S, w00, w01, w02. As in the case η = 1, draw a graph that shows the
    # epoch number vs the number of misclassifications.
    # (k) Repeat the same experiment with η = 0.1. Do not change w0, w1, w2,
    # S, w00, w01, w02. As in the case η = 1, draw a graph that shows the
    # epoch number vs the number of misclassifications.

    # i. Use the training parameter η = 1.
    learning_rates = [1, 10, 0.1]

    w0_init = np.random.uniform(-1, 1, 1)
    w1_init = np.random.uniform(-1, 1, 1)
    w2_init = np.random.uniform(-1, 1, 1)

    print('Initial weights: w0={}, w1={}, w2={}'.format(w0_init, w1_init, w2_init))

    for lr in learning_rates:
        print('------Learning Rate η = {}------'.format(lr))
        # ii. Pick w0prime, w1prime, w2prime independently and uniformly at
        # random on [−1, 1]. Write them in your report.
        # print('Initial weights: w0={}, w1={}, w2={}'.format(w0_init, w1_init, w2_init))
        w0_p = w0_init
        w1_p = w1_init
        w2_p = w2_init

        flag = True
        num_epoch = 0
        array_misclassified = []

        while flag:
            # iii.Record the number of misclassifications.
            num_misclassified = 0
            # iv.After one epoch of the perceptron training algorithm, you
            # will find a new set of weights
            w0_pp = w0_p
            w1_pp = w1_p
            w2_pp = w2_p
            # print('Current weights: w0={} w1={} w2={}'.format(w0_p, w1_p, w2_p))
            # vi.Do another epoch of the perceptron training algorithm,
            # find a new set of weights, record the number of
            # misclassifications, and so on, until convergence.
            for i in range(N):
                output = step(w0_p + w1_p * S[i][0] + w2_p * S[i][1])
                # if output == 1 and Label[i] == 0:
                #     num_misclassified += 1
                #     w0_pp = w0_pp - lr * 1
                #     w1_pp = w1_pp - lr * S[i][0]
                #     w2_pp = w2_pp - lr * S[i][1]
                # elif output == 0 and Label[i] == 1:
                #     num_misclassified += 1
                #     w0_pp = w0_pp + lr * 1
                #     w1_pp = w1_pp + lr * S[i][0]
                #     w2_pp = w2_pp + lr * S[i][1]
                if output != Label[i]:
                    num_misclassified += 1
                    w0_pp = w0_pp + lr * 1 * (Label[i] - output)
                    w1_pp = w1_pp + lr * S[i][0] * (Label[i] - output)
                    w2_pp = w2_pp + lr * S[i][1] * (Label[i] - output)

            num_epoch += 1
            w0_p = w0_pp
            w1_p = w1_pp
            w2_p = w2_pp
            # v.Record the number of misclassifications
            array_misclassified.append(num_misclassified)
            if num_misclassified == 0:
                flag = False

        # vii.Write down the final weights you obtain in your report. How
        # does these weights compare to the “optimal” weights[w0, w1, w2]?
        print('Final weights: w0\'={} w1\'={} w\'={}'.format(w0_p, w1_p, w2_p))
        print('Epoch Number: ', num_epoch)

        # (i) Regarding the previous step, draw a graph that shows the epoch
        # number vs the number of misclassifications.
        array_epoch = list(range(num_epoch))
        plt.figure()
        plt.title('Epochs vs. Misclassifications')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassifications')
        plt.plot(array_epoch, array_misclassified)
        plt.show()
