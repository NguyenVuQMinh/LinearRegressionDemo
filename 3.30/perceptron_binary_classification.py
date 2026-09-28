import pandas as pd
import numpy as np

# 1. Đọc dữ liệu
data = pd.read_csv(
    r"E:/LapTrinh/redo python 2026/LinearRegression/3.30/perceptron_student_pass_100.csv"
)

X = data[["StudyHours", "Attendance"]].values
y = data["Result"].values


# 2. Chia dữ liệu: 80% train, 20% test
np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = int(0.8 * len(X))

train = indices[:train_size]
test = indices[train_size:]

X_train = X[train]
y_train = y[train]

X_test = X[test]
y_test = y[test]


# 3. Chuẩn hóa dữ liệu
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# 4. Khởi tạo Perceptron
w = np.zeros(2)
b = 0

learning_rate = 0.01
epochs = 100


# 5. Huấn luyện
for epoch in range(epochs):

    for x, target in zip(X_train, y_train):

        # Tính đầu ra
        z = np.dot(x, w) + b

        # Hàm bước
        if z >= 0:
            prediction = 1
        else:
            prediction = 0

        # Nếu dự đoán sai → cập nhật
        error = target - prediction

        w = w + learning_rate * error * x
        b = b + learning_rate * error


# 6. Dự đoán tập test
y_pred = []

for x in X_test:

    z = np.dot(x, w) + b

    if z >= 0:
        prediction = 1
    else:
        prediction = 0

    y_pred.append(prediction)

y_pred = np.array(y_pred)


# 7. Tính TP, TN, FP, FN
TP = 0
TN = 0
FP = 0
FN = 0

for actual, prediction in zip(y_test, y_pred):

    if actual == 1 and prediction == 1:
        TP += 1

    elif actual == 0 and prediction == 0:
        TN += 1

    elif actual == 0 and prediction == 1:
        FP += 1

    elif actual == 1 and prediction == 0:
        FN += 1


# 8. Tính các độ đo
accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP) if TP + FP != 0 else 0

recall = TP / (TP + FN) if TP + FN != 0 else 0

f1 = 2 * precision * recall / (precision + recall) \
     if precision + recall != 0 else 0


# 9. In kết quả
print("Weights:", w)
print("Bias:", b)

print("\nConfusion Matrix")
print("TN =", TN, "FP =", FP)
print("FN =", FN, "TP =", TP)

print("\nEvaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)