"""Bài 5.24 - CART hồi quy với nhiều độ sâu.

Mỗi lá dự đoán trung bình các giá trị trong lá; tăng độ sâu có thể làm
mô hình bám nhiễu (overfit).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":
    # Cố định seed để dữ liệu và kết quả lặp lại được.
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + np.sin(x) + np.random.normal(0, 0.5, 50)
    plt.scatter(x, y, color="black", label="Dữ liệu")
    x_line = np.linspace(0, 10, 300).reshape(-1, 1)
    for depth, color in [(2, "red"), (5, "green"), (10, "blue")]:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(x[:, None], y)
        plt.plot(x_line, model.predict(x_line), color=color, label=f"depth={depth}")
    # Lá cây dự đoán bằng trung bình; cây quá sâu học thuộc nhiễu nên dễ overfit.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("c5/cart_regression.png", dpi=120)
    plt.close()
    print("Đã lưu biểu đồ c5/cart_regression.png")
