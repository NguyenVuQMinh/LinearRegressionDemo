import pandas as pd



data = pd.read_csv('house_price_dataset.csv')


#Ham giam do doc Gradient Descent
def gradient_descent(m1_now,m2_now, m3_now,m4_now,b_now,points,L):
    m1_gradient = 0
    m2_gradient = 0
    m3_gradient = 0
    m4_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x1 = points.iloc[i]["Area_m2"]
        x2 = points.iloc[i]["Bedrooms"]
        x3 = points.iloc[i]["House_Age_Years"]
        x4 = points.iloc[i]["Distance_to_City_Center_km"]
        y = points.iloc[i]["Price_Billion_VND"]

        prediction = (m1_now * x1 + m2_now * x2 + m3_now * x3 + m4_now * x4 + b_now)

        error = y - prediction

        m1_gradient += -(2 / n) * x1 * error
        m2_gradient += -(2 / n) * x2 * error
        m3_gradient += -(2 / n) * x3 * error
        m4_gradient += -(2 / n) * x4 * error
        b_gradient += -(2 / n) * error

    m1 = m1_now - m1_gradient * L
    m2 = m2_now - m2_gradient * L
    m3 = m3_now - m3_gradient * L
    m4 = m4_now - m4_gradient * L
    b = b_now - b_gradient * L

    return m1, m2, m3, m4, b

#Ham du doan
def predict(m1, m2, m3, m4, b, x1, x2, x3, x4):
    return m1 * x1 + m2 * x2 + m3 * x3 + m4 * x4 + b

#Main
m1 = 0
m2 = 0
m3 = 0
m4 = 0
b = 0

L = 0.000001
epochs = 2500

for i in range(epochs):
    if i % 500 == 0:
        print(f"So lan duyet qua tap du lieu: {i}")

    m1, m2, m3, m4, b = gradient_descent( m1, m2, m3, m4, b, data, L)

print(f"m1 = {m1}")
print(f"m2 = {m2}")
print(f"m3 = {m3}")
print(f"m4 = {m4}")
print(f"b = {b}")

print(f"Duong thang du doan: price = {m1} * area + {m2} * bedrooms + {m3} * age + {m4} * distance + {b}"
)


print("-----Dự đoán-----")


x1 = float(input("Nhap vao dien tich m2: "))
x2 = float(input("Nhap vao so phong: "))
x3 = float(input("Nhap vao tuoi can nha: "))
x4 = float(input("Nhap vao khoang cach den trung tam km: "))

gia = predict(m1, m2, m3, m4, b, x1, x2, x3, x4)

print(f"Gia du doan la: {gia:.2f} ti VND")