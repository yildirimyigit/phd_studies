"""
  author: yigit.yildirim@boun.edu.tr

  Note: The math and the code belongs to me but I borrowed some ideas from Ryan Harris'
  explanations on YouTube: https://www.youtube.com/watch?v=XqRUHEeiyCs
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d


def sum_error(y, y_hat):
    diff = y_hat - y
    return diff, np.sum(diff**2)


def sigm(x, der=False):
    if not der:
        return 1/(1+np.exp(-x))
    else:
        sa = sigm(x)
        return sa * (1 - sa)


def tanh(x, der=False):
    if not der:
        n = np.exp(2*x)
        return (n-1)/(n+1)
    else:
        t = tanh(x)
        return 1 - np.power(t, 2)


def elu(x, der=False):
    ind = np.where(x <= 0)
    if not der:
        x[ind] = np.exp(x[ind])-1
    else:
        x[ind] = np.exp(x[ind])
        x[x > 0] = 1
    return x


def relu(x, der=False):
    if not der:
        return np.maximum(x, 0)
    else:
        x[x < 0] = 0
        x[x >= 0] = 1
        return x


def linear(x, der=False):
    if not der:
        return x
    else:
        return np.ones([len(x), 1])


def gaussian(x, der=False):
    if not der:
        return np.exp(-x**2)
    else:
        return -2 * x * np.exp(-x ** 2)


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


class MyNN:
    """
    Creates and initializes the neural network

    @:param nn_arch: Array containing the number of nodes in each layer excluding the bias nodes.
    @:param weights: List of 2D arrays to represent all weights of the network. For each layer, we keep a matrix.
    Bias augmented.
    @:param acts: Activation functions; if not specified, the default is sigm
    """
    layer_inputs = []
    layer_outputs = []
    last_weight_deltas = []

    def __init__(self, nn_arch=None, weights=[], acts=np.empty(0)):
        if len(weights) is 0 and nn_arch is None:
            raise Exception('Cannot create a NN without any information about the architecture. Exiting...')

        if len(weights) is 0:
            self.shape = nn_arch
            self.nof_layers = len(nn_arch)  # including the input layer
            self.weights = []
            for (l1, l2) in zip(nn_arch[:-1], nn_arch[1:]):
                self.weights.append(np.random.uniform(high=0.01, size=(l1+1, l2)))  # +1 for augmenting bias
                # self.weights.append(np.random.normal(scale=0.2, size=(l1 + 1, l2)))  # +1 for augmenting the bias
                self.last_weight_deltas.append(np.zeros((l1+1, l2)))

        else:  # TODO: shape calculation missing for now
            self.nof_layers = len(weights) + 1
            self.weights = weights

        if len(acts) is 0:
            self.activations = []
            for i in range(self.nof_layers - 1):
                self.activations.append(sigm)
        else:  # TODO: dim check
            self.activations = acts

    def forward(self, x):   # assuming dimensionality match btw each x and (w0-1), where -1 represents the bias node
        self.layer_inputs = []
        self.layer_outputs = []

        layer = x
        for i in range(self.nof_layers - 1):
            b = np.ones((len(layer), 1))
            layer = np.hstack((layer, b))
            layer = np.dot(layer, self.weights[i])

            self.layer_inputs.append(layer)
            layer = self.activations[i](layer)

            self.layer_outputs.append(layer)

        return layer

    def forward_batch(self, x):   # assuming dimensionality match btw each x and (w0-1), where -1 represents bias node
        self.layer_inputs = []
        self.layer_outputs = []

        layer = x
        for i in range(self.nof_layers - 1):
            b = np.ones((len(layer), 1))
            layer = np.hstack((layer, b))
            layer = np.dot(layer, self.weights[i])

            self.layer_inputs.append(layer)
            layer = self.activations[i](layer)

            self.layer_outputs.append(layer)

        return layer

    def backprop(self, x, y, lr, momentum):
        batch_size = len(x)
        deltas = []
        y_hat = self.layer_outputs[-1]
        diff, total_error = sum_error(y, y_hat)
        for i in reversed(range(self.nof_layers-1)):
            if i == self.nof_layers - 2:  # output of the last layer
                deltas.append(diff * self.activations[i](y_hat, der=True))  # diff of each instance in the batch
                # is multiplied by the corresponding derivative
            else:
                last_delta = deltas[-1]
                delta_contributions = np.dot(last_delta, self.weights[i+1].T)
                deltas.append(delta_contributions[:, :-1]*self.activations[i](self.layer_outputs[i], der=True))

        deltas.reverse()    # to simplify index manipulation

        # here we are multiplying outputs with deltas to calculate the weight deltas
        for i in range(self.nof_layers-1):
            if i == 0:  # input layer
                this_layer_output = np.append(x, np.ones((batch_size, 1)), axis=1)
            else:
                this_layer_output = np.append(self.layer_outputs[i-1], np.ones((batch_size, 1)), axis=1)

            weight_delta = np.dot(this_layer_output.T, deltas[i]) / batch_size
            weight_delta_momentum = lr * weight_delta + momentum * self.last_weight_deltas[i]

            self.weights[i] -= weight_delta_momentum
            self.last_weight_deltas[i] = weight_delta_momentum

        return total_error

    def backprop_diff(self, diff, x, y_hat, lr=0.1, momentum=0.9):
        """ diff: error that is calculated outside"""
        diff = np.reshape(diff, (len(y_hat), 1))
        batch_size = len(x)
        deltas = []
        for i in reversed(range(self.nof_layers-1)):
            if i == self.nof_layers - 2:  # output of the last layer
                deltas.append(diff*self.activations[i](y_hat, der=True))  # diff of each instance in the batch
                # is multiplied by the corresponding derivative
            else:
                last_delta = deltas[-1]
                delta_contributions = np.dot(last_delta, self.weights[i+1].T)
                deltas.append(delta_contributions[:, :-1]*self.activations[i](self.layer_outputs[i], der=True))

        deltas.reverse()    # to simplify index manipulation

        # here we are multiplying outputs with deltas to calculate the weight deltas
        for i in range(self.nof_layers-1):
            if i == 0:  # input layer
                this_layer_output = np.append(x, np.ones((batch_size, 1)), axis=1)
            else:
                this_layer_output = np.append(self.layer_outputs[i-1], np.ones((batch_size, 1)), axis=1)

            weight_delta = np.dot(this_layer_output.T, deltas[i]) / batch_size
            weight_delta_momentum = lr * weight_delta + momentum * self.last_weight_deltas[i]

            self.weights[i] -= weight_delta_momentum
            self.last_weight_deltas[i] = weight_delta_momentum

    def epoch(self, x, y, lr=0.15, momentum=0.5):
        self.forward(x)
        err = self.backprop(x, y, lr, momentum)
        return err

    '''
    Assigns given weights to network weights. 
    (Might be needed in the outer algorithm that uses this NN.)
    '''
    def set_weights(self, w):
        self.weights = w


if __name__ == "__main__":
    neu = MyNN(nn_arch=(1, 32, 32, 1), acts=[gaussian, sigm, tanh, linear])

    x = np.linspace(0, 1, 1000)[:, None]
    y = 0.2 + 0.4*x**2 + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)

    err = []
    epochs = 1000000
    err_trsh = 1e-5
    lr = 0.2
    decay = 0.00000025
    momentum = 0.5
    for i in range(epochs):
        curr_err = neu.epoch(x, y, lr=lr, momentum=momentum)
        lr = np.maximum(lr - decay, 0.0005)
        momentum = np.maximum(momentum - decay, 0)
        err.append(curr_err)
        if i % 1000 == 0:
            print('Iteration {0}\tError: {1:0.6f}'.format(i, curr_err))
            for idx in range(neu.nof_layers-1):
                print('{0}: {1}'.format(idx, np.max(neu.weights[idx])))
        if curr_err < err_trsh:
            print('Min err in iteration {0}'.format(i))
            break

    y_hat = neu.forward(x)

    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.plot(err, c='r')
    plt.show()

    plt.xlabel('x')
    plt.ylabel('y')
    c1, = plt.plot(x, y, c='b')
    c2, = plt.plot(x, y_hat, c='r')
    plt.legend([c1, c2], ['Base Values', 'Learned Function'], loc=2)
    plt.show()

