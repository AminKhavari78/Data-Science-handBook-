# Feature Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

vec = DictVectorizer(sparse=False, dtype=int)
print(vec.fit_transform(data))
print(vec.get_feature_names())


sample = ['problem of evil',
          'evil queen',
         'horizon problem']

text = CountVectorizer()
X = text.fit_transform(sample)
print(X)
print(pd.DataFrame(X.toarray(), columns=text.get_feature_names()))

vek = TfidfVectorizer()
Y = vek.fit_transform(sample)
print(pd.DataFrame(Y.toarray(), columns=vek.get_feature_names()))


x1 = np.array([1, 2, 3, 4, 5])
y1 = np.array([4, 2, 1, 3, 7])
X1 = x1[:, np.newaxis]
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X1)
print(X2)

model = LinearRegression().fit(X2, y1)
yfit = model.predict(X2)
plt.scatter(x1, y1)
plt.plot(x1, yfit);
plt.show()

model = make_pipeline(SimpleImputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
model.fit(X1, y1) # X with missing values, from above
print(y1)
print(model.predict(X1))

