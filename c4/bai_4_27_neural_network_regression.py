"""Bài 4.27 - Hồi quy bằng mạng nơ-ron NumPy."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bai_4_26_neural_network_numpy import NeuralNetwork


if __name__ == "__main__":
    # Chuẩn hóa đặc trưng và mục tiêu giúp gradient có thang đo ổn định.
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    y_mean, y_std = y_train.mean(), y_train.std()
    model = NeuralNetwork((24, 12), 0.01, 5000, task="regression")
    model.fit(X_train, (y_train - y_mean) / y_std)
    pred = model.predict(X_test) * y_std + y_mean
    # R2 = 1 - SSE/SST; NSE = 1 - sum((y-p)^2)/sum((y-mean(y))^2).
    # MAE = mean(|y-p|); RMSE = sqrt(mean((y-p)^2)).
    r2 = r2_score(y_test, pred)
    nse = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"R2={r2:.4f} | NSE={nse:.4f} | MAE={mean_absolute_error(y_test, pred):.4f}"
          f" | RMSE={np.sqrt(mean_squared_error(y_test, pred)):.4f}")
