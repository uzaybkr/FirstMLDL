# FirstMLDL

Machine learning and deep learning **from scratch** in pure NumPy. No PyTorch, no TensorFlow, no scikit-learn.
The code starts with a straight line fitted by gradient descent and ends with a neural network that reads handwritten digits.

Built while working through *Programming Machine Learning* by Paolo Perrotta, and inspired by [PyMLDL](https://github.com/Attilaali/PyMLDL).

## Results on MNIST

| Model | Test accuracy | Training time (CPU) |
|---|---|---|
| Softmax regression (1 linear layer) | **92.23%** | ~5 s |
| Neural network (128 ReLU hidden units) | **97.55%** | ~18 s |

Both use 10 epochs, learning rate 0.1, mini-batches of 64, and seed 0.

## Quickstart

```bash
git clone https://github.com/<your-username>/FirstMLDL.git
cd FirstMLDL
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

pytest                          # 5 tests, ~2 s
firstmldl nn                    # downloads MNIST (~11 MB, checksum-verified) on first run
firstmldl softmax --epochs 20
firstmldl nn --hidden 256 --lr 0.05
```

## Use it as a library

```python
from firstmldl import NeuralNetwork, accuracy
from firstmldl.data import load_mnist

X_train, y_train, X_test, y_test = load_mnist()
model = NeuralNetwork(hidden=128, epochs=10).fit(X_train, y_train, verbose=True)
print(accuracy(model.predict(X_test), y_test))
```

## The learning path

| Step | Model | Idea it teaches |
|---|---|---|
| 1 | `LinearRegression` | Loss functions and gradient descent |
| 2 | `SoftmaxRegression` | Classification, softmax, cross-entropy, mini-batches |
| 3 | `NeuralNetwork` | Hidden layers, ReLU, backpropagation, weight initialization |

`tests/test_models.py` checks that each step works:
- linear regression recovers a known line
- softmax regression separates clusters
- the neural network learns XOR, which a linear model can't
- backpropagation matches a numerical gradient estimate

## Project layout

```
firstmldl/
  models.py     all models + the shared mini-batch training loop
  data.py       MNIST download, checksum, preprocessing
  __main__.py   command-line interface
tests/          pytest suite (runs in CI on every push)
```

## Add your own model

Every classifier inherits the training loop from `Classifier`. To add a new model, define three methods:

```python
class MyModel(Classifier):
    def init_params(self, n_in, n_out): ...   # dict of numpy arrays
    def predict_proba(self, X): ...           # (n, n_classes) probabilities
    def gradients(self, X, Y): ...            # dict of gradients, same keys as params
```

Add it to the gradient-check test, and the math is verified for you.

## Roadmap

- [ ] Save and load trained weights
- [ ] Convolutional network
- [ ] Personalized models trained on our own data

## License

MIT
