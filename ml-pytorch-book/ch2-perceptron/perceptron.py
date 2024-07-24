import numpy as np


class Perceptron:
    """
    This is the original perceptron proposed by Rosenblatt. Notice
    this implementation is fully linear, which means that non-linear
    activation functions (e.g.: sigmoid) are used. This is basically
    a linear combination of parameters with a simple threshold activation
    for predicting.

    This network is very simple and developed to deal with binary classification
    problems, where the output belongs to one of two classes (0 or 1).
    However, it could be extended to multi-class prediction using the One-versus-All
    (OvA) technique, where on classifier (Perceptron) is trained for each class you
    want to classify, and the all classifiers are combined for the multiclass problem.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta  # Learning rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # Random draws numbers from uniform distribution
        rgen = np.random.RandomState(self.random_state)

        # Size of w_ (weights) is basically initialized to match the number
        # of features in X. Weights are initialized std of 0.01. Initializing
        # weights randomly, in this case, helps the learning rate to have effect
        # on the decision boundary. If weights were initialized to zero the eta
        # would affect the weights vector scale but not the direction.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # self.w_ = np.zeros(X.shape[1])  # To test how it doesn't affect decision boundary

        # Bias is a scalar
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(0, self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # This is the original perceptron learning rule
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input
        Notice that bias (as per the original idea) was to helps the network
        learn the threshold activation easily, by subtracting or adding to the
        weight summation in net_input the amount necessary for classification.

        Uses numpy vectorization instead of for loops to multiply the matrices,
        which is more efficient and benefits when training on GPUs and on CPUs
        using the single-instruction multiple-data (SIMD) support. To achieve
        vectorization numpy uses highly optimized C-libraries for linear algebra
        like Basic Linear Algebra Subprograms (BLAS) and Linear Algebra Package
        (LAPACK).
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
