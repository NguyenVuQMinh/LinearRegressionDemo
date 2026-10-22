"""Bài 5.25 - Tạo CSV và xây pipeline cây quyết định.

Ví dụ minh họa toàn bộ quy trình: sinh dữ liệu, lưu/đọc CSV, chia tập,
huấn luyện cây phân lớp và xem độ quan trọng của đặc trưng.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


if __name__ == "__main__":
    # Dữ liệu giả lập có quy tắc nhãn rõ ràng để minh họa pipeline.
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "tuoi": rng.integers(18, 65, n),
        "thu_nhap": rng.integers(5, 40, n),
        "diem_tin_dung": rng.integers(300, 850, n),
    })
    df["mua_hang"] = ((df["thu_nhap"] > 18) & (df["diem_tin_dung"] > 600)).astype(int)
    csv_path = Path(__file__).with_name("khach_hang_gia_lap.csv")
    df.to_csv(csv_path, index=False)
    data = pd.read_csv(csv_path)
    # Tách cột mục tiêu khỏi X trước khi train.
    # Nếu có cột phân loại, cần encode bằng get_dummies trước bước này.
    X = data.drop(columns="mua_hang")
    y = data["mua_hang"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    for leaf in [1, 5, 10]:
        model = DecisionTreeClassifier(
            min_samples_leaf=leaf, random_state=42
        ).fit(X_train, y_train)
        print(
            f"min_samples_leaf={leaf}, "
            f"accuracy={accuracy_score(y_test, model.predict(X_test)):.3f}"
        )
    print("Feature importances (tổng = 1, số lớn hơn quan trọng hơn):")
    print(dict(zip(X.columns, model.feature_importances_)))
    plt.figure(figsize=(11, 6))
    plot_tree(model, feature_names=list(X.columns), class_names=["Không", "Có"], filled=True)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("tree.png"), dpi=120)
    plt.close()
    # Nếu mục tiêu là số liên tục, có thể thay classifier bằng DecisionTreeRegressor.
    reg = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
    print("Ví dụ cây hồi quy, dự đoán 3 mẫu:", reg.predict(X_test.iloc[:3]))
