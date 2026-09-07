# Du doan gia nha bang hoi quy tuyen tinh

Chuong trinh su dung **Linear Regression** va thuat toan **Gradient Descent** de du doan gia nha dua tren:

- `area`: dien tich nha, don vi m2
- `room`: so phong
- `price`: gia nha, don vi **ti VNĐ**

## Cau truc project

```text
dataset_gia_nha.csv   # Du lieu huan luyen, gom 100 dong
linearregression.py   # Chuong trinh huan luyen va du doan
```

## Cai dat

Can Python 3 va thu vien pandas:

```bash
pip install pandas matplotlib
```



## Chay chuong trinh

Mo terminal tai thu muc project va chay:

```bash
python linearregression.py
```

Sau khi mo hinh hoc xong, nhap dien tich va so phong, vi du:

```text
Nhap vao dien tich: 100
Nhap vao so phong: 3
Gia du doan la: 3.08 ti vnđ
```

## Mo hinh

Mo hinh co dang:

```text
price = m1 * area + m2 * room + b
```

Trong do, `m1`, `m2` va `b` duoc cap nhat qua nhieu epoch bang Gradient Descent.

```python
L = 0.000001
epochs = 2500
```

## Luu y

- File CSV phai duoc dat cung thu muc voi `linearregression.py`.
- Cot CSV can giu dung ten `area`, `room`, `price`.
- Gia trong dataset dang o don vi ti dong. Vi du `1.65` nghia la 1.65 ti dong.
