"""Bài 1: Ước lượng khoảng tin cậy bootstrap cho trung bình."""

import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.array([58, 70, 82, 95, 63, 77, 88, 69, 73, 85])
    rng = np.random.default_rng(42)

    # Bootstrap lấy mẫu có hoàn lại, giữ nguyên kích thước mẫu gốc.
    means = np.array(
        [rng.choice(data, size=len(data), replace=True).mean() for _ in range(2000)]
    )
    # Phân vị 2.5% và 97.5% tạo khoảng tin cậy xấp xỉ 95%.
    confidence_interval = np.percentile(means, [2.5, 97.5])
    print("mean=", data.mean(), "CI 95%=", confidence_interval)

    plt.hist(means, bins=35, alpha=0.75)
    plt.axvline(data.mean(), color="red", label="mean gốc")
    plt.axvline(confidence_interval[0], color="black", ls="--")
    plt.axvline(
        confidence_interval[1],
        color="black",
        ls="--",
        label="CI 95%",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
