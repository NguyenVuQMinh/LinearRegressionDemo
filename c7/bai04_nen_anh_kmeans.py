"""Bài 4: Nén ảnh bằng cách gom màu với K-Means."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def main():
    # Tự sinh ảnh màu để bài chạy được mà không cần tải thêm dữ liệu.
    rng = np.random.default_rng(42)
    height, width = 120, 160
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: height // 2, :, :] = [70, 150, 220]
    image[height // 2 :, :, :] = [30, 170, 70]
    for _ in range(20):
        y, x = rng.integers(0, height), rng.integers(0, width)
        image[max(0, y - 8) : y + 8, max(0, x - 8) : x + 8] = rng.integers(
            0, 256, 3
        )

    # Mỗi pixel RGB là một điểm 3 chiều; reshape lại sau khi fit.
    pixels = image.reshape(-1, 3).astype(float) / 255
    model = KMeans(n_clusters=16, n_init=5, random_state=42).fit(pixels)
    compressed = model.cluster_centers_[model.labels_].reshape(height, width, 3)
    print(
        f"Anh {height}x{width}: mau ly thuyet 256^3, con 16 mau; "
        f"ti le con {16 / (256 ** 3):.8f}"
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Goc")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(compressed)
    plt.title("Nen 16 mau")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
