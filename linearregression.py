import pandas as pd



data = pd.read_csv('dataset_gia_nha.csv')


#Ham giam do doc Gradient Descent
def gradient_descent(m1_now,m2_now,b_now,points,L):
    m1_gradient = 0
    m2_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x1 = points.iloc[i].area
        x2 = points.iloc[i].room
        y = points.iloc[i].price

        m1_gradient += -(2/n) * x1 * (y - (m1_now * x1 + m2_now * x2 + b_now))
        m2_gradient += -(2/n) * x2 * (y - (m1_now * x1 + m2_now * x2 + b_now))
        b_gradient += -(2/n) * (y - (m1_now * x1 + m2_now * x2 + b_now))

    m1 = m1_now - m1_gradient * L
    m2 = m2_now - m2_gradient * L
    b = b_now - b_gradient * L

    return m1,m2,b

#Ham du doan
def predict(m1,m2,b,x1_p,x2_p):
    return  (m1 * x1_p) + (m2 * x2_p) + b

#Main
m1 = 0
m2 = 0
b = 0
L = 0.000001
epochs = 2500

for i in range(epochs):
    if i % 500 == 0:
        print(f"Số lần duyệt qua tập dữ liệu: {i}")
    m1,m2,b = gradient_descent(m1,m2,b,data,L)

print(f"m1 = {m1},m2 = {m2},b = {b}")
print(f"Đường thẳng dự đoán : price = {m1} * area + {m2} * room + {b}")


print("-----Dự đoán-----")
x1 = int(input("Nhập vào diện tích (m2): "))
x2 = int(input("Nhập vào số phòng: "))
gia = predict(m1,m2,b,x1,x2)
print(f"Giá dự đoán là: {gia:.2f} tỉ vnđ")