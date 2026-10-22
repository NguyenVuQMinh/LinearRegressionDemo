"""Bài 8: Khám phá ảnh hưởng của eps và minPts trong DBSCAN."""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles


def main():
    X, _ = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
    settings = [(0.2, 5), (0.5, 5), (1.0, 5), (0.5, 10)]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))

    for axis, (eps, min_points) in zip(axes.ravel(), settings):
        labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(X)
        axis.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=8)
        axis.set_title(f"eps={eps}, minPts={min_points}")
        clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(
            eps,
            min_points,
            "clusters=",
            clusters,
            "noise=",
            sum(labels == -1),
        )

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
