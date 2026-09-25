import numpy as np
import pytest

from firstmldl.models import LinearRegression, NeuralNetwork, SoftmaxRegression, accuracy, one_hot

rng = np.random.default_rng(42)


def blobs(n=300):
    centers = np.array([[0, 0], [4, 4], [-4, 4]])
    y = np.arange(n) % 3
    return centers[y] + rng.normal(size=(n, 2)), y


def xor(n=400):
    X = rng.uniform(-1, 1, (n, 2))
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)


def test_linear_regression_recovers_line():
    X = rng.uniform(-1, 1, (200, 2))
    y = X @ np.array([3.0, -2.0]) + 0.5
    model = LinearRegression(lr=0.1, epochs=2000).fit(X, y)
    np.testing.assert_allclose(model.w, [3.0, -2.0], atol=1e-3)
    assert model.b == pytest.approx(0.5, abs=1e-3)


def test_softmax_separates_blobs():
    X, y = blobs()
    model = SoftmaxRegression(lr=0.1, epochs=50).fit(X, y)
    assert accuracy(model.predict(X), y) > 0.95


def test_neural_network_learns_xor_where_linear_model_cannot():
    X, y = xor()
    linear = SoftmaxRegression(lr=0.1, epochs=200).fit(X, y)
    net = NeuralNetwork(hidden=16, lr=0.5, epochs=300, batch_size=32).fit(X, y)
    assert accuracy(linear.predict(X), y) < 0.7
    assert accuracy(net.predict(X), y) > 0.95


def test_saved_weights_reload_to_identical_predictions(tmp_path):
    X, y = blobs()
    model = NeuralNetwork(hidden=8, epochs=5).fit(X, y)
    path = tmp_path / "weights" / "nn"
    model.save(path)
    np.testing.assert_array_equal(NeuralNetwork.load(path).predict_proba(X), model.predict_proba(X))


@pytest.mark.parametrize("model", [SoftmaxRegression(), NeuralNetwork(hidden=5)], ids=["softmax", "nn"])
def test_backprop_matches_numerical_gradient(model):
    X = rng.normal(size=(8, 4))
    Y = one_hot(np.arange(8) % 3, 3)
    model.params = model.init_params(4, 3)
    for p in model.params.values():
        p += rng.normal(scale=0.1, size=p.shape)
    grads = model.gradients(X, Y)

    eps = 1e-6
    for name, p in model.params.items():
        numeric = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            original = p[idx]
            p[idx] = original + eps
            up = model.loss(X, Y)
            p[idx] = original - eps
            down = model.loss(X, Y)
            p[idx] = original
            numeric[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grads[name], numeric, rtol=1e-4, atol=1e-7)
