# Hyperparameters and Model Validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
"""
print(X)
print(y)
"""

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
y_model = model.predict(X)
print(accuracy_score(y, y_model))

X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                  train_size=0.5)
model.fit(X1, y1)
y2_model = model.predict(X2)
print(accuracy_score(y2, y2_model))

print(cross_val_score(model, X, y, cv=5))

scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))
print(scores)