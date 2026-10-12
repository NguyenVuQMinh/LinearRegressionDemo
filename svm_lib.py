from __future__ import print_function
import numpy as np 
from sklearn.svm import SVC
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
X = np.concatenate((X0, X1), axis=0) # all data
y = np.concatenate((np.ones(N), -np.ones(N))) # labels

# Train a linear SVM using scikit-learn.
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

w = model.coef_.ravel()
b = model.intercept_[0]
support_vectors = model.support_vectors_

decision_values = model.decision_function(model.support_vectors_)
distances = np.abs(decision_values) / np.linalg.norm(model.coef_)

print(distances)

print('support vectors = ')
print(support_vectors)
print('w = ', w)
print('b = ', b)