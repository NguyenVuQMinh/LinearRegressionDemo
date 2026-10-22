"""Bài 3.30 - Dùng Perceptron tự cài đặt trên dữ liệu ung thư vú."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bai_3_29_perceptron import Perceptron


if __name__ == "__main__":
    # Tách dữ liệu trước khi chuẩn hóa để tránh rò rỉ thông tin từ tập kiểm tra.
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Perceptron dùng -1/+1, còn dataset sklearn dùng 0/1.
    y_train_pm = np.where(y_train == 1, 1, -1)
    model = Perceptron(learning_rate=0.001, n_iters=1000).fit(X_train, y_train_pm)
    y_pred = np.where(model.predict(X_test) == 1, 1, 0)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # Precision: trong các mẫu dự đoán dương, tỷ lệ thật sự dương.
    # Recall: trong các mẫu dương thật, tỷ lệ được phát hiện đúng.
