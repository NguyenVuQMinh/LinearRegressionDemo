"""Bài 5: So sánh khởi tạo random và k-means++."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main():
    X, _ = make_blobs(
        n_samples=600,
        centers=[(-6, 0), (0, 8), (7, -2)],
        cluster_std=[(0.5, 3), (0.6, 2.5), (0.5, 3)],
        random_state=42,
    )
    results = {}
    for init in ("random", "k-means++"):
        results[init] = [
            KMeans(
                n_clusters=3, init=init, n_init=1, random_state=seed
            ).fit(X).inertia_
            for seed in range(20)
        ]
        print(
            init,
            "mean inertia=",
            round(np.mean(results[init]), 2),
            "min=",
            round(min(results[init]), 2),
        )

    all_values = results["random"] + results["k-means++"]
    edges = np.linspace(min(all_values) - 1, max(all_values) + 1, 11)
    plt.hist(results["random"], bins=edges, alpha=0.6, label="random")
    plt.hist(results["k-means++"], bins=edges, alpha=0.6, label="k-means++")
    plt.xlabel("Inertia")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
