"""Bài 5: So sánh các mô hình ensemble."""

import time

from sklearn.datasets import make_classification
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    models = {
        "Bagging": BaggingClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(
            f"{name:18s} accuracy={model.score(X_test, y_test):.3f}, "
            f"time={elapsed:.4f}s"
        )
    # Bagging giảm variance và có thể chạy song song; Boosting tuần tự để giảm bias.


if __name__ == "__main__":
    main()
