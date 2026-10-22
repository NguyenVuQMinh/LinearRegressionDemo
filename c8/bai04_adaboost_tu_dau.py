"""Bài 4: Tự cài đặt AdaBoost với decision stump."""

import numpy as np


class DecisionStump:
    """Cây quyết định một mức, chỉ xét đặc trưng đầu tiên."""

    def fit(self, X, y, weights):
        best = (float("inf"), None, None, None)
        values = np.unique(X[:, 0])
        thresholds = (values[:-1] + values[1:]) / 2
        for threshold in thresholds:
            for polarity in (1, -1):
                predictions = np.where(
                    polarity * X[:, 0] < polarity * threshold,
                    -1,
                    1,
                )
                error = np.sum(weights[predictions != y])
                if error < best[0]:
                    best = (error, threshold, polarity, predictions)
        self.error, self.threshold, self.polarity, _ = best
        return self

    def predict(self, X):
        return np.where(
            self.polarity * X[:, 0] < self.polarity * self.threshold,
            -1,
            1,
        )


def main():
    X = np.arange(1, 9).reshape(-1, 1)
    y = np.array([-1, -1, -1, 1, 1, 1, 1, 1])
    weights = np.ones(len(y)) / len(y)
    models = []

    for iteration in range(3):
        stump = DecisionStump().fit(X, y, weights)
        error = max(stump.error, 1e-10)  # Tránh chia cho 0 khi error bằng 0.
        if error > 0.5:
            break
        # alpha = 1/2 * log((1-error)/error) là trọng số của stump.
        alpha = 0.5 * np.log((1 - error) / error)
        predictions = stump.predict(X)
        weights *= np.exp(-alpha * y * predictions)
        weights /= weights.sum()
        models.append((alpha, stump))
        print(
            f"vong {iteration + 1}: alpha={alpha:.4f}, "
            f"threshold={stump.threshold:.2f}, polarity={stump.polarity}"
        )

    score = sum(alpha * stump.predict(X) for alpha, stump in models)
    print("du doan ensemble:", np.where(score >= 0, 1, -1))
    print("trong so:", weights)


if __name__ == "__main__":
    main()
