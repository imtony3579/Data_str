import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style



data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predice = "G3"

x = np.array(data.drop([predice], 1))
y = np.array(data[predice])
x_train, x_test , y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

#a = True
#while a:
#    x_train, x_test , y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)
#
#    # Below commented code is to save the model in a file using pickle
#
#
#    linear = linear_model.LinearRegression()
#    linear.fit(x_train, y_train)
#
#    acc = linear.score(x_test,y_test)
#
#    if acc >= 0.93:
#        with open("student.pickle", "wb") as f:
#            pickle.dump(linear, f)
#        print(acc)
#        a = False


pickle_in = open("student.pickle", "rb")
linear = pickle.load(pickle_in)

predice = linear.predict(x_test)

#for i in range(len(predice)):
#    print(int(round(predice[i])), y_test[i])

p = "G2"
style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("final")
plt.show()

