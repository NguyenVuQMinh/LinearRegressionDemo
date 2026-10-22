"""Bài 3: Chọn số cụm bằng phương pháp Elbow."""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main():
    X, _ = make_blobs(n_samples=500, centers=5, random_state=42)
    values = []
    for k in range(1, 11):
        # Inertia là WCSS; n_init=10 giảm rủi ro rơi vào cực tiểu địa phương.
        values.append(
            KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_
        )

    print("inertia:", [round(value, 1) for value in values])
    print("Elbow kỳ vọng: K=5")
    plt.plot(range(1, 11), values, "o-")
    plt.xlabel("K")
    plt.ylabel("Inertia (WCSS)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
