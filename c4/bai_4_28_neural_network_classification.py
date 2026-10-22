"""Bài 4.28 - Phân lớp nhị phân bằng mạng nơ-ron NumPy."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bai_4_26_neural_network_numpy import NeuralNetwork


if __name__ == "__main__":
    # Giữ tỷ lệ lớp khi tách dữ liệu để phép đánh giá đáng tin cậy hơn.
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Chỉ fit scaler trên train; test chỉ được transform để tránh leakage.
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    # Bài này là nhị phân nên output dùng sigmoid và y được encode 0/1.
    # Nếu là đa lớp, output cần softmax và nhãn phải encode phù hợp.
    model = NeuralNetwork((20, 10), 0.01, 2500, task="classification")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1-score:", f1_score(y_test, pred))
