"""Bài 5.23 - CART phân lớp với scikit-learn.

CART dùng chỉ số Gini để chọn phép chia; ``max_depth`` giúp minh họa
đánh đổi giữa khả năng học và nguy cơ overfit.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


if __name__ == "__main__":
    # Tách phân tầng để tỷ lệ ba lớp Iris được giữ ở train và test.
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # random_state giúp tái lập kết quả; cây quá sâu thường dễ overfit.
    for depth in [None, 2, 3]:
        model = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        print(f"max_depth={depth}: train={accuracy_score(y_train, model.predict(X_train)):.3f},"
              f" test={accuracy_score(y_test, model.predict(X_test)):.3f}")
    model = DecisionTreeClassifier(
        criterion="gini", random_state=42
    ).fit(X_train, y_train)
    plt.figure(figsize=(12, 7))
    plot_tree(model, feature_names=load_iris().feature_names,
              class_names=load_iris().target_names, filled=True)
    plt.tight_layout()
    plt.savefig("c5/cart_iris_tree.png", dpi=120)
    plt.close()
    print("Các feature được xét tại nút:", load_iris().feature_names)
