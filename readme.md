# Du doan gia nha bang hoi quy tuyen tinh

Chuong trinh su dung **Linear Regression** va thuat toan **Gradient Descent** de du doan gia nha dua tren 4 dac trung:

- `Area_m2`: dien tich nha, don vi m2
- `Bedrooms`: so phong ngu
- `House_Age_Years`: tuoi can nha, don vi nam
- `Distance_to_City_Center_km`: khoang cach den trung tam thanh pho, don vi km

Gia nha duoc luu trong cot `Price_Billion_VND`, don vi **ti VND**.

## Cau truc project

```text
house_price_dataset.csv # Du lieu huan luyen, gom 100 dong
linearregression.py   # Chuong trinh huan luyen va du doan
```

## Cai dat

Can Python 3 va thu vien pandas:

```bash
pip install pandas
```



## Chay chuong trinh

Mo terminal tai thu muc project va chay:

```bash
python linearregression.py
```

Sau khi mo hinh hoc xong, nhap dien tich, so phong, tuoi can nha va khoang cach den trung tam. Vi du:

```text
Nhap vao dien tich m2: 100
Nhap vao so phong: 3
Nhap vao tuoi can nha: 5
Nhap vao khoang cach den trung tam km: 4
Gia du doan la: 5.00 ti VND
```

## Mo hinh

Mo hinh co dang:

```text
price = m1 * area + m2 * bedrooms + m3 * age + m4 * distance + b
```

Trong do, cac he so `m1`, `m2`, `m3`, `m4` va `b` duoc cap nhat qua nhieu epoch bang Gradient Descent.

```python
L = 0.000001
epochs = 2500
```

## Luu y

- File CSV phai duoc dat cung thu muc voi `linearregression.py`.
- Dataset gom 100 dong du lieu va 5 cot.
- Ten cot CSV can giu dung:
	`Area_m2`, `Bedrooms`, `House_Age_Years`, `Distance_to_City_Center_km`, `Price_Billion_VND`.
- Gia trong dataset dang o don vi ti dong. Vi du `2.1` nghia la 2.1 ti dong.
