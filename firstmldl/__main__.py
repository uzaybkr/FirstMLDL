import argparse
import time

from .data import load_mnist
from .models import NeuralNetwork, SoftmaxRegression, accuracy


def main(argv=None):
    parser = argparse.ArgumentParser(prog="firstmldl", description="Train a from-scratch model on MNIST.")
    parser.add_argument("model", choices=["softmax", "nn"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128, help="hidden units (nn only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    X_train, y_train, X_test, y_test = load_mnist()
    common = dict(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    model = NeuralNetwork(hidden=args.hidden, **common) if args.model == "nn" else SoftmaxRegression(**common)

    start = time.perf_counter()
    model.fit(X_train, y_train, verbose=True)
    print(f"trained in {time.perf_counter() - start:.1f}s")
    print(f"train accuracy {accuracy(model.predict(X_train), y_train):.2%}")
    print(f"test accuracy  {accuracy(model.predict(X_test), y_test):.2%}")


if __name__ == "__main__":
    main()
