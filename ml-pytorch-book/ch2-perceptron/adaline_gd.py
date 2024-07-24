import numpy as np


class AdalineGD:
    """ADAptive LInear NEuron classifier
    This is the algorithm proposed by Bernard Widrow and Tedd Hoff.
    This algorithm is very similar to the Perceptron with exception
    that in this case we use an activation function to compute how
    much delta need to be updated on the weights. The benefit of
    using an activation function is that it allows to use gradient
    descent algorithm for optimization.

    This algorithm trains using full-batch training data, which means
    the whole dataset is evaluated to compute the gradient descent
    step every epoch. For large datasets this makes the learning much
    slower.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Notice the introduction of activation compared to Perceptron
            output = self.activation(net_input)
            errors = (y - output)
            # This is the gradient descent from the mean squared error loss function
            # Differently from Perceptron, the gradient descent calculate the weights
            # update over all training samples
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_  # Notice use of broadcasting for bias

    def activation(self, X):
        """Compute linear activation, this example uses simple identity function"""
        return X

    def predict(self, X):
        """Return class label after unit step classifier"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
