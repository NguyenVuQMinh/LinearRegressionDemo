from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


train_data = pd.read_csv(Path(__file__).with_name('house_price_dataset.csv'))
val_data   = pd.read_csv(Path(__file__).with_name('house_price_dataset_validate.csv'))
test_data  = pd.read_csv(Path(__file__).with_name('house_price_dataset_test.csv'))



print(f"So dong train: {len(train_data)}")
print(f"So dong validate: {len(val_data)}")
print(f"So dong test: {len(test_data)}")

print(train_data.head())


# ===== Bước 2: Chuẩn hóa dữ liệu (z-score) =====
# QUAN TRỌNG: mean và std chỉ được tính trên tập train,
# sau đó dùng chung cho val và test (không tính lại)

x_mean = train_data["x_dien_tich_m2"].mean()
x_std  = train_data["x_dien_tich_m2"].std()

print(f"\nx_mean (train) = {x_mean}")
print(f"x_std  (train) = {x_std}")

def normalize_x(x, mean, std):
    return (x - mean) / std

# Tạo cột x đã chuẩn hóa cho cả 3 tập
train_data["x_norm"] = normalize_x(train_data["x_dien_tich_m2"], x_mean, x_std)
val_data["x_norm"]   = normalize_x(val_data["x_dien_tich_m2"], x_mean, x_std)
test_data["x_norm"]  = normalize_x(test_data["x_dien_tich_m2"], x_mean, x_std)

print("\n--- Sau khi chuan hoa ---")
print(train_data[["x_dien_tich_m2", "x_norm"]].head())

# Kiem tra: x_norm cua train phai co mean ~ 0, std ~ 1
print(f"\nKiem tra train x_norm: mean={train_data['x_norm'].mean():.4f}, "
      f"std={train_data['x_norm'].std():.4f}")

# ===== Bước 3: Xây dựng feature đa thức + Gradient Descent overfitting =====

DEGREE = 10  # bậc đa thức cao -> dễ overfitting với ít dữ liệu

def build_polynomial_features(x, degree):
    """
    Tra ve ma tran cac dac trung [x, x^2, x^3, ..., x^degree]
    x: pandas Series hoac numpy array (da chuan hoa)
    """
    x = np.array(x)
    n = len(x)
    X_poly = np.zeros((n, degree))
    for d in range(1, degree + 1):
        X_poly[:, d - 1] = x ** d
    return X_poly


def scale_polynomial_features(X_train, X_other):
    """Scale polynomial columns using only statistics from the train set."""
    feature_scales = np.std(X_train, axis=0)
    feature_scales[feature_scales == 0] = 1
    return X_train / feature_scales, X_other / feature_scales, feature_scales


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def predict_poly(X_poly, m, b):
    """
    X_poly: ma tran (n, degree)
    m: vector he so (degree,)
    """
    return X_poly @ m + b


def gradient_descent_poly(X_poly, y, m, b, L):
    n = len(y)

    prediction = predict_poly(X_poly, m, b)
    error = y - prediction

    # Gradient cho tung he so m_j (vector hoa thay vi vong lap)
    m_gradient = -(2 / n) * (X_poly.T @ error)
    b_gradient = -(2 / n) * np.sum(error)

    m_new = m - L * m_gradient
    b_new = b - L * b_gradient

    return m_new, b_new


# Chuan bi du lieu dang numpy
X_train_poly_raw = build_polynomial_features(train_data["x_norm"], DEGREE)
y_train = train_data["y_gia_nha_ty"].to_numpy()

X_val_poly_raw = build_polynomial_features(val_data["x_norm"], DEGREE)
y_val = val_data["y_gia_nha_ty"].to_numpy()

X_train_poly, X_val_poly, overfit_scales = scale_polynomial_features(
    X_train_poly_raw, X_val_poly_raw
)

# Khoi tao he so
m = np.zeros(DEGREE)
b = 0.0

L = 0.01
epochs = 3000

train_losses = []
val_losses = []

for epoch in range(epochs):
    m, b = gradient_descent_poly(X_train_poly, y_train, m, b, L)

    # Tinh loss tren ca train va validate sau moi epoch
    train_pred = predict_poly(X_train_poly, m, b)
    val_pred = predict_poly(X_val_poly, m, b)

    train_mse = compute_mse(y_train, train_pred)
    val_mse = compute_mse(y_val, val_pred)

    train_losses.append(train_mse)
    val_losses.append(val_mse)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: train_mse={train_mse:.4f}, val_mse={val_mse:.4f}")

print(f"\nFinal train MSE: {train_losses[-1]:.4f}")
print(f"Final val MSE:   {val_losses[-1]:.4f}")


# ===== Bước 4: Trực quan hóa overfitting =====

# 4a. Loss curve: train vs validate theo epoch
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"Loss Curve (Overfitting) - Degree = {DEGREE}")
plt.legend()
plt.grid(True)
plt.show()

# 4b. Scatter du lieu + duong cong du doan
x_range = np.linspace(train_data["x_norm"].min(), train_data["x_norm"].max(), 300)
X_range_poly_raw = build_polynomial_features(x_range, DEGREE)
X_range_poly = X_range_poly_raw / overfit_scales
y_range_pred = predict_poly(X_range_poly, m, b)

plt.figure(figsize=(8, 5))
plt.scatter(train_data["x_norm"], train_data["y_gia_nha_ty"],
            color="blue", label="Train data", alpha=0.6)
plt.scatter(val_data["x_norm"], val_data["y_gia_nha_ty"],
            color="orange", label="Validate data", alpha=0.6)
plt.plot(x_range, y_range_pred, color="red", label=f"Fitted curve (degree={DEGREE})")
plt.xlabel("x_norm (dien tich da chuan hoa)")
plt.ylabel("y_gia_nha_ty")
plt.title("Duong cong du doan - Overfitting")
plt.legend()
plt.grid(True)
plt.show()

# ----- 5a. Cách 1: Thử nhiều bậc đa thức, chọn bậc có val_loss thấp nhất -----

def train_polynomial(degree, X_train_poly, y_train, X_val_poly, y_val,
                      L, epochs, lam=0.0):
    """
    Train mot mo hinh polynomial regression.
    lam: he so regularization L2 (Ridge). lam=0 -> khong regularization.
    Tra ve: m, b, train_losses, val_losses
    """
    m = np.zeros(degree)
    b = 0.0
    train_losses = []
    val_losses = []

    n = len(y_train)

    for epoch in range(epochs):
        prediction = predict_poly(X_train_poly, m, b)
        error = y_train - prediction

        # Gradient co them so hang regularization L2: + (2*lam/n)*m
        m_gradient = -(2 / n) * (X_train_poly.T @ error) + (2 * lam / n) * m
        b_gradient = -(2 / n) * np.sum(error)

        m = m - L * m_gradient
        b = b - L * b_gradient

        train_mse = compute_mse(y_train, predict_poly(X_train_poly, m, b))
        val_mse = compute_mse(y_val, predict_poly(X_val_poly, m, b))

        train_losses.append(train_mse)
        val_losses.append(val_mse)

    return m, b, train_losses, val_losses


print("=== Thu nghiem nhieu bac da thuc (chon degree tot nhat theo validate) ===")

degrees_to_try = range(1, 11)
L_search = 0.01
epochs_search = 3000

results = {}

for d in degrees_to_try:
    X_tr_raw = build_polynomial_features(train_data["x_norm"], d)
    X_va_raw = build_polynomial_features(val_data["x_norm"], d)
    X_tr, X_va, degree_scales = scale_polynomial_features(X_tr_raw, X_va_raw)

    m_d, b_d, tr_losses, va_losses = train_polynomial(
        d, X_tr, y_train, X_va, y_val, L_search, epochs_search
    )

    final_val_loss = va_losses[-1]
    results[d] = {
        "m": m_d, "b": b_d,
        "train_losses": tr_losses, "val_losses": va_losses,
        "final_val_loss": final_val_loss,
        "scales": degree_scales
    }

    print(f"Degree {d:2d}: final_train_mse={tr_losses[-1]:.4f}, "
          f"final_val_mse={final_val_loss:.4f}")

# Chon degree co val loss thap nhat
best_degree = min(results, key=lambda d: results[d]["final_val_loss"])
print(f"\n>>> Degree tot nhat theo validate set: {best_degree}")

GOOD_DEGREE = best_degree
good_result = results[GOOD_DEGREE]
good_m = good_result["m"]
good_b = good_result["b"]
good_train_losses = good_result["train_losses"]
good_val_losses = good_result["val_losses"]
good_scales = good_result["scales"]

X_test_poly_overfit_raw = build_polynomial_features(test_data["x_norm"], DEGREE)
X_test_poly_overfit = X_test_poly_overfit_raw / overfit_scales
y_test = test_data["y_gia_nha_ty"].to_numpy()

X_test_poly_good_raw = build_polynomial_features(test_data["x_norm"], GOOD_DEGREE)
X_test_poly_good = X_test_poly_good_raw / good_scales

test_mse_overfit = compute_mse(y_test, predict_poly(X_test_poly_overfit, m, b))
test_mse_good = compute_mse(
    y_test, predict_poly(X_test_poly_good, good_m, good_b)
)

print("\n===== KET QUA CUOI CUNG TREN TEST SET =====")
print(f"Mo hinh OVERFIT (degree={DEGREE}):     test_mse = {test_mse_overfit:.4f}")
print(f"Mo hinh GOOD FIT (degree={GOOD_DEGREE}): test_mse = {test_mse_good:.4f}")


# ===== Bước 7: So sánh trực quan tổng kết =====

# 7a. So sanh Loss Curve: Overfit vs Good Fit
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses, label="Validation Loss")
axes[0].set_title(f"OVERFIT - Degree = {DEGREE}")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(good_train_losses, label="Train Loss")
axes[1].plot(good_val_losses, label="Validation Loss")
axes[1].set_title(f"GOOD FIT - Degree = {GOOD_DEGREE}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 7b. So sanh duong cong du doan: Overfit vs Good Fit
x_range = np.linspace(train_data["x_norm"].min(), train_data["x_norm"].max(), 300)

X_range_overfit_raw = build_polynomial_features(x_range, DEGREE)
X_range_overfit = X_range_overfit_raw / overfit_scales
y_range_overfit = predict_poly(X_range_overfit, m, b)

X_range_good_raw = build_polynomial_features(x_range, GOOD_DEGREE)
X_range_good = X_range_good_raw / good_scales
y_range_good = predict_poly(X_range_good, good_m, good_b)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_range_pred, title in [
    (axes[0], y_range_overfit, f"OVERFIT - Degree = {DEGREE}"),
    (axes[1], y_range_good, f"GOOD FIT - Degree = {GOOD_DEGREE}"),
]:
    ax.scatter(train_data["x_norm"], train_data["y_gia_nha_ty"],
               color="blue", label="Train", alpha=0.6)
    ax.scatter(val_data["x_norm"], val_data["y_gia_nha_ty"],
               color="orange", label="Validate", alpha=0.6)
    ax.scatter(test_data["x_norm"], test_data["y_gia_nha_ty"],
               color="green", label="Test", alpha=0.6)
    ax.plot(x_range, y_range_pred, color="red", label="Fitted curve")
    ax.set_title(title)
    ax.set_xlabel("x_norm")
    ax.set_ylabel("y_gia_nha_ty")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()