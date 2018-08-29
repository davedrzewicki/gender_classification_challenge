import numpy as np
#import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    ]


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


array = dict()

for i in range(0, len(names)): # loops through the classifiers that I've chosen
    clf = classifiers[i]

    tally = []
    xNew = []
    yNew = []
    x = 0

    for j in range(len(X)): # loops through each value of X matrix, calculates accuracy of all "leave one out" predictions
        test = [X[j]]
        ans = Y[j]
        xNew = list(X)
        yNew = list(Y)
        del xNew[j]
        del yNew[j]
        clf = clf.fit(xNew, yNew)
        prediction = clf.predict(test)
        value = int(prediction == ans)
        tally.append(value) # 1 if the value was correctly predicted


    x = sum(tally)/len(X) # this gives an accuracy
    array[names[i]] = '{0:.2f}'.format(x)
    

max_value = max(array.values())  # maximum value
max_keys = [k for k, v in array.items() if v == max_value] # getting all keys containing the `maximum`

print('Max value is ', max_value, 'which is obtained from ', max_keys)