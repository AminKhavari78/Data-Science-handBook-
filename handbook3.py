#Unsupervised learning: Iris clustering

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)

model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)

y_gmm = model.predict(X_iris)

X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False);

plt.show()
