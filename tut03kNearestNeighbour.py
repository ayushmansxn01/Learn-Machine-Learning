from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading data sets
iris=datasets.load_iris()
print(iris.DESCR)
features=iris.data
labels=iris.target
# print(features[0], labels[0])

#training classifier
clf=KNeighborsClassifier()
clf.fit(features, labels)

preds=clf.predict([[2,1,1,1]])
print(preds)
