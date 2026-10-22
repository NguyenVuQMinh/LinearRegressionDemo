"""Bài 5.22 - Cây ID3 từ đầu trên bảng ``Mua máy tính``.

ID3 chọn thuộc tính có Information Gain lớn nhất ở mỗi nút.
"""

import math
from collections import Counter


# Mỗi dòng gồm bốn thuộc tính và nhãn quyết định ở phần tử cuối.
DATA = [
    ["Trẻ", "Cao", "Không", "Tốt", "Không"],
    ["Trẻ", "Cao", "Không", "Xuất sắc", "Không"],
    ["Trung niên", "Cao", "Không", "Tốt", "Có"],
    ["Già", "Trung bình", "Không", "Tốt", "Có"],
    ["Già", "Thấp", "Có", "Tốt", "Có"],
    ["Già", "Thấp", "Có", "Xuất sắc", "Không"],
    ["Trung niên", "Thấp", "Có", "Xuất sắc", "Có"],
    ["Trẻ", "Trung bình", "Không", "Tốt", "Không"],
    ["Trẻ", "Thấp", "Có", "Tốt", "Có"],
    ["Già", "Trung bình", "Có", "Tốt", "Có"],
    ["Trẻ", "Trung bình", "Có", "Xuất sắc", "Có"],
    ["Trung niên", "Trung bình", "Không", "Xuất sắc", "Có"],
    ["Trung niên", "Cao", "Có", "Tốt", "Có"], ["Già", "Trung bình", "Không", "Xuất sắc", "Không"],
]
FEATURES = ["Tuổi", "Thu nhập", "Sinh viên", "Đánh giá tín dụng"]


def entropy(y):
    """Tính entropy: ``H(S) = -sum(p_i * log2(p_i))``."""
    n = len(y)
    # H(S) = -sum(p_i*log2(p_i)); bỏ qua p_i=0 để tránh log(0).
    return -sum((count / n) * math.log2(count / n)
                for count in Counter(y).values() if count)


def information_gain(X, y, feature):
    """Tính độ giảm entropy khi chia theo một thuộc tính."""
    base = entropy(y)
    values = {row[feature] for row in X}
    weighted = sum(
        len(sub_y) / len(y) * entropy(sub_y)
        for value in values
        for sub_y in [[target for row, target in zip(X, y) if row[feature] == value]]
    )
    # Gain(S,A) = H(S) - sum(|S_v|/|S|)*H(S_v).
    return base - weighted


def build_tree(X, y, features):
    """Đệ quy xây cây; lá là nhãn phổ biến nhất khi không thể chia tiếp."""
    if len(set(y)) == 1:
        return y[0]
    if not features:
        return Counter(y).most_common(1)[0][0]
    gains = [information_gain(X, y, f) for f in features]
    best_pos = max(range(len(features)), key=gains.__getitem__)
    best = features[best_pos]
    if gains[best_pos] <= 0:  # Điều kiện dừng khi Information Gain bằng 0.
        return Counter(y).most_common(1)[0][0]
    tree = {best: {}}
    remaining = features[:best_pos] + features[best_pos + 1:]
    for value in {row[best] for row in X}:
        sub = [(row, target) for row, target in zip(X, y) if row[best] == value]
        tree[best][value] = build_tree([r for r, _ in sub], [t for _, t in sub], remaining)
    return tree


def predict(tree, sample):
    """Đi theo các nhánh của cây để dự đoán một mẫu."""
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    branch = tree[feature].get(sample[feature])
    return "Không biết" if branch is None else predict(branch, sample)


if __name__ == "__main__":
    # Bốn cột đầu là đặc trưng, cột cuối là nhãn ``Mua máy tính``.
    X = [row[:4] for row in DATA]
    y = [row[4] for row in DATA]
    tree = build_tree(X, y, list(range(4)))
    print("Cây ID3:", tree)
    for sample in [["Trẻ", "Trung bình", "Có", "Tốt"],
                   ["Già", "Cao", "Không", "Xuất sắc"],
                   ["Trung niên", "Thấp", "Có", "Tốt"]]:
        print(sample, "=>", predict(tree, sample))
