from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 
from scipy.optimize import minimize
# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = V.T.dot(V) # see definition of V, K near eq (8)

p = -np.ones(2*N) # all-one vector
# build A, b, G, h 
# Solve the dual problem without the optional cvxopt dependency.
constraints = ({'type': 'eq', 'fun': lambda lam: y.ravel().dot(lam)},)
result = minimize(
	lambda lam: 0.5 * lam.dot(K).dot(lam) + p.dot(lam),
	np.zeros(2 * N),
	jac=lambda lam: K.dot(lam) + p,
	bounds=[(0, None)] * (2 * N),
	constraints=constraints,
	method='SLSQP',
	options={'ftol': 1e-10, 'maxiter': 1000},
)
if not result.success:
	raise RuntimeError(f'SVM optimization failed: {result.message}')

l = result.x.reshape(-1, 1)
print('lambda = ')
print(l.T)
epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)