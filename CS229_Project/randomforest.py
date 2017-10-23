import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from nolearn.dbn import DBN
import timeit


# Load the data
X_train = pd.read_csv('train.csv', header=None)
X_train = np.array(X_train)
X_test = pd.read_csv('test.csv', header=None)
X_test = np.array(X_test)

y_train = [0,1,2,3,4]*(len(X_train)/5)
y_test = [0,1,2,3,4]*(len(X_test)/5)

clf_rf = RandomForestClassifier(criterion='entropy', n_estimators=100, min_samples_split=2, n_jobs=-1,max_depth=10)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print "random forest accuracy: ",acc_rf

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train, y_train)
y_pred_sgd = clf_sgd.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print "stochastic gradient descent accuracy: ",acc_sgd


clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print "Linear SVM accuracy: ",acc_svm


clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print "nearest neighbors accuracy: ",acc_knn




