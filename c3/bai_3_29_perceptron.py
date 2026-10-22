"""Bài 3.29 - Cài đặt Perceptron từ đầu bằng NumPy.

Perceptron là bộ phân lớp tuyến tính: mỗi mẫu được gán nhãn dựa trên
dấu của ``w.x + b`` và chỉ cập nhật tham số khi dự đoán sai.
"""

import numpy as np


class Perceptron:
    """Bộ phân lớp Perceptron cho nhãn ``-1`` và ``+1``."""

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        """Huấn luyện bằng quy tắc cập nhật Perceptron."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        # Nhãn phải là -1 và +1, không dùng 0/1 vì y=0 sẽ không cập nhật.
        if not np.all(np.isin(y, (-1, 1))):
            raise ValueError("Perceptron yêu cầu nhãn -1 hoặc +1")
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        # n_iters là số lượt quét dữ liệu; nhiều lượt giúp hội tụ khi cần.
        for _ in range(self.n_iters):
            errors = 0
            for xi, target in zip(X, y):
                if target * (np.dot(xi, self.weights) + self.bias) <= 0:
                    # Quy tắc: w <- w + eta*y*x; bias cũng được cập nhật.
                    self.weights += self.learning_rate * target * xi
                    self.bias += self.learning_rate * target
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        """Dự đoán nhãn theo dấu của điểm tuyến tính."""
        scores = np.dot(np.asarray(X), self.weights) + self.bias
        return np.where(scores >= 0, 1, -1)


if __name__ == "__main__":
    # Tạo hai cụm dữ liệu tuyến tính để minh họa khả năng hội tụ.
    rng = np.random.default_rng(42)
    X = np.vstack((rng.normal((-2, -2), 0.7, (30, 2)),
                   rng.normal((2, 2), 0.7, (30, 2))))
    y = np.r_[-np.ones(30, dtype=int), np.ones(30, dtype=int)]
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    print("Trọng số:", model.weights, "bias:", model.bias)
    print("Accuracy:", np.mean(model.predict(X) == y))
