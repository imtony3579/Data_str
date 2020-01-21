import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from matplotlib import style
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
#print(data.head())


# Now most of the data is non-numerical 
# so to convert those data we will be using 
# preprocessor of sklear 

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = 'class'

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

model = KNeighborsClassifier(n_neighbors = 7)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("predicted : ", names[predicted[x]], "   Actual : ", names[y_test[x]])



