"""Bài 4: So sánh SVM tuyến tính và SVM với kernel RBF."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        stratify=data.target,
        random_state=42,
    )

    for C in (0.1, 1, 10):
        model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))
        model.fit(Xtr, ytr)
        print(
            f"linear C={C}: support={model[-1].n_support_.sum()}, "
            f"accuracy={model.score(Xte, yte):.3f}"
        )

    for C, gamma in ((1, "scale"), (10, 0.01), (10, 0.1)):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))
        model.fit(Xtr, ytr)
        print(
            f"RBF C={C}, gamma={gamma}: "
            f"accuracy={model.score(Xte, yte):.3f}"
        )

    # PCA chỉ dùng để vẽ; không dùng để huấn luyện vì làm mất thông tin.
    Z = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(data.data)
    )
    clf = SVC(kernel="rbf", C=1, gamma="scale").fit(Z, data.target)
    xx, yy = np.meshgrid(
        np.linspace(Z[:, 0].min() - 1, Z[:, 0].max() + 1, 250),
        np.linspace(Z[:, 1].min() - 1, Z[:, 1].max() + 1, 250),
    )
    predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, predictions, alpha=0.2)
    plt.scatter(Z[:, 0], Z[:, 1], c=data.target, s=8)
    plt.title("SVM RBF trên mặt phẳng PCA")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
