import preprocess
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib

def train_classifier(clf, X_test, y_test):
	score = clf.score(X_test, y_test)
	print score