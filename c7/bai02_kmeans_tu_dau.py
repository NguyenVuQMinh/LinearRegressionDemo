"""Bài 2: Tự cài đặt K-Means chỉ bằng NumPy."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler


class MyKMeans:
    """K-Means với khởi tạo tâm ngẫu nhiên và cập nhật theo trung bình."""

    def __init__(self, K, max_iters=100, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        self.centers_ = X[rng.choice(len(X), self.K, replace=False)].copy()

        for _ in range(self.max_iters):
            old = self.centers_.copy()
            labels = self.predict(X)
            for k in range(self.K):
                # Gán điểm vào tâm gần nhất rồi cập nhật tâm bằng trung bình.
                points = X[labels == k]
                self.centers_[k] = points.mean(axis=0) if len(points) else old[k]
            if np.allclose(old, self.centers_):
                break

        self.labels_ = self.predict(X)
        self.inertia_ = sum(
            np.sum((X[self.labels_ == k] - self.centers_[k]) ** 2)
            for k in range(self.K)
        )
        return self

    def predict(self, X):
        """Trả về tâm gần nhất theo khoảng cách Euclid bình phương."""
        distances = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)


def main():
    data = load_iris()
    X = StandardScaler().fit_transform(data.data[:, :2])
    model = MyKMeans(3).fit(X)
    print(
        "inertia=",
        round(model.inertia_, 3),
        "Rand Index=",
        round(rand_score(data.target, model.labels_), 3),
    )
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis")
    plt.scatter(
        model.centers_[:, 0],
        model.centers_[:, 1],
        c="red",
        marker="x",
        s=100,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
