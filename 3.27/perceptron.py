import numpy as np

w = np.array([[1], [2], [-10]])
x = np.array([[3], [4], [1]])

if(w.T @ x == -1):
    print("True")
else:
    print("False")