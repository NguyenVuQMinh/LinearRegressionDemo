"""Bài 7: Tự cài đặt DBSCAN và đối chiếu với scikit-learn."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


class MyDBSCAN:
    """DBSCAN dựa trên ma trận khoảng cách Euclid."""

    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.X_ = X
        distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
        self.labels_ = np.full(len(X), -1)
        visited = np.zeros(len(X), bool)
        cluster = 0
        neighbors = [
            np.flatnonzero(distances[i] <= self.eps).tolist() for i in range(len(X))
        ]
        core = np.array([len(items) >= self.min_samples for items in neighbors])
        self.core_sample_indices_ = np.flatnonzero(core)

        for i in range(len(X)):
            if visited[i]:
                continue
            visited[i] = True
            if not core[i]:
                continue  # Có thể trở thành điểm biên khi gặp cụm sau đó.

            self.labels_[i] = cluster
            seeds = list(neighbors[i])
            j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True
                    if core[q]:
                        seeds.extend(
                            item for item in neighbors[q] if item not in seeds
                        )
                if self.labels_[q] == -1:
                    self.labels_[q] = cluster
                j += 1
            cluster += 1
        return self


def main():
    X, _ = make_blobs(
        n_samples=350, centers=3, cluster_std=0.55, random_state=42
    )
    rng = np.random.default_rng(42)
    X = np.vstack([X, rng.uniform(-6, 10, (25, 2))])
    X = StandardScaler().fit_transform(X)
    mine = MyDBSCAN(0.25, 5).fit(X)
    sklearn_model = DBSCAN(eps=0.25, min_samples=5).fit(X)
    mine_clusters = len(set(mine.labels_)) - (1 if -1 in mine.labels_ else 0)
    sklearn_clusters = len(set(sklearn_model.labels_)) - (
        1 if -1 in sklearn_model.labels_ else 0
    )
    print(
        "So cum tu tao:",
        mine_clusters,
        "Sklearn:",
        sklearn_clusters,
        "noise:",
        sum(mine.labels_ == -1),
    )
    plt.scatter(X[:, 0], X[:, 1], c=mine.labels_, cmap="tab10")
    plt.title("MyDBSCAN (-1 la noise)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
