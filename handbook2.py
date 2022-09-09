#Unsupervised learning example: Iris dimensionality

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

X_iris = iris.drop('species', axis=1)

model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);

plt.show()