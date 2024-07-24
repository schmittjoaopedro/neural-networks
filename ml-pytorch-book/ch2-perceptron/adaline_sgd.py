import numpy as np


class AdalineSGD:
    """Differently from AdalineGD algorithm this version implements
    Stochastic Gradient Descent, which differs in the sense that not
    all data samples are applied every epoch to compute the gradient
    step, but instead it takes one gradient step for each sample.
    This makes the learning faster, specially for larger datasets,
    but the convergence noisier.

    SGD also allows for partial learning (or online learning) which is
    the ability of further improving the model as new data comes in, like
    for example in online websites.

    However, to make sure the algorithm is converging, we calculate an
    average loss after each epoch, but this is only for gathering stats.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=1):
        self.eta = eta  # Learning rate
        self.n_iter = n_iter
        self.w_initialized = False  # used to disable weights reset on online-learning
        self.shuffle = shuffle  # shuffle training dataset to avoid cyclic learning
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        # Gradient descent step, based on partial derivatives for each weight and bias
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
