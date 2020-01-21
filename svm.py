import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm
import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2)

# print(x_train, y_train)
classes = ['maligment', 'benign']

cls = svm.SVC(kernel="linear", C = 2)
cls.fit(x_train,y_train)

y_predict = cls.predict(x_test)

acc = metrics.accuracy_score(y_test, y_predict)
print(acc)



