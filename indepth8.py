# Kernel SVM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
#from indepth7 import plot_svc_decision_function


X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)



def plot_3D(elev=30, azim=30, X=X, y=y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = X[:, 0]
    y = X[:, 1]
    r = np.exp(-(X ** 2).sum(1))

    ax.scatter(x, y, r)
    ax.set_title('for amin khavari')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()





