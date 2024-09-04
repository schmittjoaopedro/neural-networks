import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes

        # hidden
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer
        # input dim:  [n_examples, n_features]
        #       dot   [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim:  [n_examples, n_hidden]
        #        dot  [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        ########################
        # Output layer weights
        ########################

        # one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        # = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeights
        # where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        # for convenient reuse

        # input/output dim: [n_examples, n_classes]
        d_loss__d_out_act = 2. * (a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_out_act__d_out_net = a_out * (1. - a_out)  # Sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_out_act * d_out_act__d_out_net

        # gradient for output weights

        # [n_examples, n_hidden]
        d_out_net__d_out_weights = a_h

        # input dim:  [n_classes, n_examples]
        #        dot  [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__d_out_weights = np.dot(delta_out.T, d_out_net__d_out_weights)
        d_loss__d_out_bias = np.sum(delta_out, axis=0)

        # Part 2: dLoss/dHiddenWeights
        # = deltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet
        #   * dHiddenNet/dHiddenWeights

        # [n_classes, n_hidden]
        d_out_net__d_hidden_act = self.weight_out

        # output dim: [n_examples, n_hidden]
        d_loss__d_hidden_act = np.dot(delta_out, d_out_net__d_hidden_act)

        # [n_examples, n_hidden]
        d_hidden_act__d_hidden_net = a_h * (1. - a_h)  # sigmoid derivate

        # [n_examples, n_features]
        d_hidden_net__d_hidden_weights = x

        # output dim: [n_hidden, n_features]
        d_loss_d_hidden_weights = np.dot((d_loss__d_hidden_act * d_hidden_act__d_hidden_net).T,
                                         d_hidden_net__d_hidden_weights)
        d_loss_d_hidden_bias = np.sum((d_loss__d_hidden_act * d_hidden_act__d_hidden_net), axis=0)

        return (d_loss__d_out_weights, d_loss__d_out_bias,
                d_loss_d_hidden_weights, d_loss_d_hidden_bias)
