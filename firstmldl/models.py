import numpy as np


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class LinearRegression:
    """Fit y ≈ Xw + b with full-batch gradient descent on mean squared error."""

    def __init__(self, lr=0.01, epochs=1000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(self.epochs):
            err = self.predict(X) - y
            self.w -= self.lr * 2 * X.T @ err / len(X)
            self.b -= self.lr * 2 * err.mean()
        return self

    def predict(self, X):
        return X @ self.w + self.b


class Classifier:
    """Mini-batch gradient descent on cross-entropy.

    Subclasses define init_params, predict_proba and gradients; the training loop is shared.
    """

    def __init__(self, lr=0.1, epochs=10, batch_size=64, seed=0):
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, verbose=False):
        Y = one_hot(y, y.max() + 1)
        self.params = self.init_params(X.shape[1], Y.shape[1])
        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                i = order[start:start + self.batch_size]
                for name, grad in self.gradients(X[i], Y[i]).items():
                    self.params[name] -= self.lr * grad
            if verbose:
                print(f"epoch {epoch:>3}/{self.epochs}  loss {self.loss(X, Y):.4f}")
        return self

    def loss(self, X, Y):
        return -np.mean(np.sum(Y * np.log(self.predict_proba(X) + 1e-12), axis=1))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class SoftmaxRegression(Classifier):
    """Multiclass logistic regression: one linear layer + softmax."""

    def init_params(self, n_in, n_out):
        return {"W": np.zeros((n_in, n_out)), "b": np.zeros(n_out)}

    def predict_proba(self, X):
        return softmax(X @ self.params["W"] + self.params["b"])

    def gradients(self, X, Y):
        g = (self.predict_proba(X) - Y) / len(X)
        return {"W": X.T @ g, "b": g.sum(axis=0)}


class NeuralNetwork(Classifier):
    """One ReLU hidden layer + softmax output, trained with backpropagation."""

    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden

    def init_params(self, n_in, n_out):
        normal = self.rng.normal
        return {
            "W1": normal(0, np.sqrt(2 / n_in), (n_in, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": normal(0, np.sqrt(1 / self.hidden), (self.hidden, n_out)),
            "b2": np.zeros(n_out),
        }

    def _forward(self, X):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0)
        return z1, h, softmax(h @ p["W2"] + p["b2"])

    def predict_proba(self, X):
        return self._forward(X)[2]

    def gradients(self, X, Y):
        z1, h, probs = self._forward(X)
        g2 = (probs - Y) / len(X)
        g1 = (g2 @ self.params["W2"].T) * (z1 > 0)
        return {"W1": X.T @ g1, "b1": g1.sum(axis=0), "W2": h.T @ g2, "b2": g2.sum(axis=0)}
