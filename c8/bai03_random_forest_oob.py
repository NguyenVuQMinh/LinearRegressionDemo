"""Bài 3: Random Forest và đánh giá OOB (out-of-bag)."""

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def main():
    X, y = load_digits(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    ).fit(X, y)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        X,
        y,
        cv=5,
        n_jobs=-1,
    )
    print(
        f"OOB score={model.oob_score_:.4f}; "
        f"CV 5 folds={scores.mean():.4f} +/- {scores.std():.4f}"
    )
    # OOB là các mẫu không nằm trong bootstrap của mỗi cây (xấp xỉ 36.8%),
    # vì vậy có thể dùng như một tập validation nội tại.


if __name__ == "__main__":
    main()
