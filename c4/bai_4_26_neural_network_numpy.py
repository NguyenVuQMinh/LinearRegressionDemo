"""Bài 4.26 - Mạng nơ-ron đơn giản, chỉ dùng NumPy.

Mạng dùng ReLU ở các tầng ẩn, sigmoid cho phân lớp nhị phân và đầu ra
tuyến tính cho hồi quy. Huấn luyện bằng lan truyền ngược theo batch đầy đủ.
"""

import numpy as np


class NeuralNetwork:
    """Mạng truyền thẳng tối giản hỗ trợ phân lớp và hồi quy."""

    def __init__(self, hidden_layer_sizes=(16,), learning_rate=0.01,
                 n_iters=1000, task="classification", random_state=42):
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.task = task
        self.rng = np.random.default_rng(random_state)
        self.weights = []
        self.biases = []

    @staticmethod
    def _relu(z):
        """Hàm kích hoạt ReLU: ``max(0, z)``."""
        return np.maximum(0, z)

    @staticmethod
    def _relu_gradient(z):
        """Đạo hàm ReLU, dùng cho bước lan truyền ngược."""
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z):
        """Sigmoid ổn định số học bằng cách giới hạn đầu vào."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _forward(self, X):
        """Tính lần lượt các kích hoạt và giá trị trước kích hoạt."""
        activations = [X]
        zs = []
        a = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            # W có dạng (số nơ-ron tầng trước, số nơ-ron tầng sau).
            z = a @ W + b
            zs.append(z)
            a = self._relu(z)
            activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        output = self._sigmoid(z) if self.task == "classification" else z
        activations.append(output)
        return activations, zs

    def fit(self, X, y):
        """Khởi tạo tham số rồi tối ưu bằng gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        sizes = [X.shape[1], *self.hidden_layer_sizes, 1]
        # Khởi tạo nhỏ, ngẫu nhiên (không phải 0) để phá vỡ đối xứng.
        self.weights = [self.rng.normal(0, 0.1, (a, b))
                        for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((1, size)) for size in sizes[1:]]
        for _ in range(self.n_iters):
            activations, zs = self._forward(X)
            # Với BCE + sigmoid, đạo hàm tại đầu ra rút gọn thành ``a - y``.
            delta = activations[-1] - y
            for layer in range(len(self.weights) - 1, -1, -1):
                grad_w = activations[layer].T @ delta / len(X)
                grad_b = np.mean(delta, axis=0, keepdims=True)
                old_w = self.weights[layer].copy()
                self.weights[layer] -= self.learning_rate * grad_w
                self.biases[layer] -= self.learning_rate * grad_b
                if layer > 0:
                    delta = (delta @ old_w.T) * self._relu_gradient(zs[layer - 1])
        return self

    def predict_proba(self, X):
        """Trả về xác suất (phân lớp) hoặc giá trị đầu ra (hồi quy)."""
        return self._forward(np.asarray(X, dtype=float))[0][-1].ravel()

    def predict(self, X):
        """Chuyển xác suất thành nhãn; hồi quy trả về giá trị liên tục."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int) if self.task == "classification" else proba


if __name__ == "__main__":
    # Ví dụ hàm ngưỡng đơn giản: số âm thuộc lớp 0, số dương thuộc lớp 1.
    X = np.array([[-2], [-1], [1], [2]], dtype=float)
    y = np.array([0, 0, 1, 1])
    net = NeuralNetwork((5,), 0.2, 2000).fit(X, y)
    print("Dự đoán:", net.predict(X))
