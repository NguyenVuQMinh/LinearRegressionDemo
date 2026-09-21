import pandas as pd
import numpy as np

#Ham dua du lieu thanh mang luy thua cho da thuc  
def create_polynomial_features(X, degree):
    features = []

    for x in X:
        row = []

        for d in range(degree + 1):
            row.append(x ** d)

        features.append(row)

    return np.array(features)

def standardize(X_train, X_test):
    # Tính mean và std từ dữ liệu Train
    mean = X_train[:, 1:].mean(axis=0)
    std = X_train[:, 1:].std(axis=0)

    # Chuẩn hóa Train
    X_train_scaled = X_train.astype(float, copy=True)
    X_train_scaled[:, 1:] = (
        X_train[:, 1:] - mean
    ) / std

    # Chuẩn hóa Test bằng mean và std của Train
    X_test_scaled = X_test.astype(float, copy=True)
    X_test_scaled[:, 1:] = (
        X_test[:, 1:] - mean
    ) / std

    return X_train_scaled, X_test_scaled

def ridge_gradient_descent(X, y, learning_rate, epochs, lambda_):
    m, n = X.shape

    w = np.zeros(n)
    loss_history = []

    for epoch in range(epochs):

        # Dự đoán
        y_pred = X @ w

        # Sai số
        error = y_pred - y

        # Ridge Loss
        loss = np.sum(error ** 2) / (2 * m) \
               + (lambda_ / 2) * np.sum(w[1:] ** 2)

        loss_history.append(loss)

        # Gradient của MSE
        gradient = (X.T @ error) / m

        # Gradient của Ridge
        # Không phạt w[0] (bias)
        gradient[1:] += lambda_ * w[1:]

        # Cập nhật trọng số
        w = w - learning_rate * gradient

    return w, loss_history

def gradient_descent(X, y, learning_rate, epochs):
    m, n = X.shape

    # Khởi tạo tất cả trọng số bằng 0
    w = np.zeros(n)

    loss_history = []

    for epoch in range(epochs):

        # 1. Dự đoán
        y_pred = X @ w

        # 2. Tính sai số
        error = y_pred - y

        # 3. Tính Loss
        loss = np.sum(error ** 2) / (2 * m)

        loss_history.append(loss)

        # 4. Tính Gradient
        gradient = (X.T @ error) / m

        # 5. Cập nhật trọng số
        w = w - learning_rate * gradient
        if epoch % 1000 == 0:
            print(epoch)

    return w, loss_history

def predict(X, w):
    return X @ w

def mse(y_true, y_pred):
    m = len(y_true)

    return np.sum((y_pred - y_true) ** 2) / m



#Main

# 1. Đọc dữ liệu
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X_train = train_data["area"].values
y_train = train_data["price"].values

X_test = test_data["area"].values
y_test = test_data["price"].values

# 2. Tạo Polynomial Features
degree = 12

# Ma trận X test sau khi nhân với x, x^2,...x^n
X_train_poly = create_polynomial_features(
    X_train, degree
)
# Ma trận X test sau khi nhân với x, x^2,...x^n
X_test_poly = create_polynomial_features(
    X_test, degree
)

# 3. Standardize, chuẩn hóa ma trận x 
X_train_scaled, X_test_scaled = standardize(
    X_train_poly,
    X_test_poly
)


# 4. Train bằng Gradient Descent
learning_rate = 0.01
epochs = 30000

w, loss_history = gradient_descent(
    X_train_scaled,
    y_train,
    learning_rate,
    epochs
)

# 5. Dự đoán
y_train_pred = predict(
    X_train_scaled,
    w
)

y_test_pred = predict(
    X_test_scaled,
    w
)

# 6. Tính Error
train_error = mse(
    y_train,
    y_train_pred
)

test_error = mse(
    y_test,
    y_test_pred
)

# 7. In kết quả
print("===== KHÔNG DÙNG RIDGE =====")
print("Train Error:", train_error)
print("Test Error:", test_error)

lambda_ = 0.2

w_ridge, ridge_loss_history = ridge_gradient_descent(
    X_train_scaled,
    y_train,
    learning_rate,
    epochs,
    lambda_
)


# ==============================
# 8. Dự đoán bằng Ridge
# ==============================

y_train_pred_ridge = predict(
    X_train_scaled,
    w_ridge
)

y_test_pred_ridge = predict(
    X_test_scaled,
    w_ridge
)


# ==============================
# 9. Tính Error của Ridge
# ==============================

train_error_ridge = mse(
    y_train,
    y_train_pred_ridge
)

test_error_ridge = mse(
    y_test,
    y_test_pred_ridge
)


print("\n===== DÙNG RIDGE =====")
print("Train Error:", train_error_ridge)
print("Test Error:", test_error_ridge)



