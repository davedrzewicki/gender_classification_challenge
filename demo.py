from sklearn import tree

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

tally = []

xNew = []
yNew = []

#for i in range(0,9):

for i in range(len(X)):
    test = [X[i]]
    ans = Y[i]
    xNew = list(X)
    yNew = list(Y)
    del xNew[i]
    del yNew[i]
    clf = clf.fit(xNew, yNew)
    prediction = clf.predict(test)
    value = int(prediction == ans)
    tally.append(value)
    #print(xNew, '\n', yNew, '\n', prediction)


print(tally)
x = sum(tally)/len(X)
print(sum(tally), 'out of', len(X), 'Accuracy:', '{0:.2f}'.format(x))


