"""Bài 2: Tự cài đặt Logistic Regression nhị phân."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    """Đưa điểm số về xác suất trong khoảng (0, 1)."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def train(X, y, lr, epochs=1000):
    """Huấn luyện bằng gradient descent và trả về lịch sử binary cross-entropy."""
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []

    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        pc = np.clip(p, 1e-9, 1 - 1e-9)  # Tránh log(0) tạo NaN.
        losses.append(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

        # Gradient theo công thức: dL/dw = mean((p-y)x), dL/db = mean(p-y).
        w -= lr * (X.T @ (p - y)) / len(y)
        b -= lr * np.mean(p - y)

    return w, b, losses


def main():
    iris = load_iris()
    mask = iris.target < 2
    X = StandardScaler().fit_transform(iris.data[mask, :2])
    y = iris.target[mask]
    results = {}

    for lr in (0.001, 0.01, 0.1):
        results[lr] = train(X, y, lr)
        print(f"lr={lr}: loss cuoi = {results[lr][2][-1]:.4f}")

    lr = 0.1
    w, b, loss = results[lr]
    plt.plot(loss, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for label in (0, 1):
        plt.scatter(
            X[y == label, 0],
            X[y == label, 1],
            label=iris.target_names[label],
        )

    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    plt.plot(xs, -(w[0] * xs + b) / w[1], "k--", label="decision boundary")
    plt.xlabel("sepal length (scaled)")
    plt.ylabel("sepal width (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
