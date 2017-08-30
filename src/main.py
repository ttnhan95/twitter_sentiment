import w2v_get_data
import fasttext
import os
import os.path
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
import  train_classifier

if os.path.isfile("../data_train/w2v.train"):
	os.remove("../data_train/w2v.train")

print "Preparing Data"
content=""
content += w2v_get_data.get_data_csv() 

#write to file data
with open("../data_train/w2v.train", "w") as text_file:
    text_file.write(content)

#train w2vmodel
print "Training W2V Model"
model = fasttext.skipgram('../data_train/w2v.train', '../model/twittermodel', dim=300)

#extract feature vector
import run_w2v
run_w2v.get_w2v("../data/twitter2013.train.txt", "../data_train/twitter2013.train")
run_w2v.get_w2v("../data/twitter2013.dev.txt", "../data_train/twitter2013.dev")
run_w2v.get_w2v("../data/twitter2013.test.txt", "../data_train/twitter2013.test")
run_w2v.get_w2v("../data/twitter2014.test.txt", "../data_train/twitter2014.test")
run_w2v.get_w2v("../data/twitter2015.train.txt", "../data_train/twitter2015.train")
run_w2v.get_w2v("../data/twitter2015.test.txt", "../data_train/twitter2015.test")
run_w2v.get_w2v("../data/twitter2016.train.txt", "../data_train/twitter2016.train")
run_w2v.get_w2v("../data/twitter2016.dev.txt", "../data_train/twitter2016.dev")
run_w2v.get_w2v("../data/twitter2016.devtest.txt", "../data_train/twitter2016.devtest")
run_w2v.get_w2v("../data/twitter2016.test.txt", "../data_train/twitter2016.test")

#train classifier
print "Loading Data ..."
X_train = []
y_train = []
X_train, y_train = preprocess.load_data("../data_train/twitter2013.train", X_train, y_train)
X_train, y_train = preprocess.load_data("../data_train/twitter2013.dev", X_train, y_train)
# X_train, y_train = preprocess.load_data("../data_train/twitter2015.train", X_train, y_train)
X_train, y_train = preprocess.load_data("../data_train/twitter2016.dev", X_train, y_train)
X_train, y_train = preprocess.load_data("../data_train/twitter2016.train", X_train, y_train)

X_test_2013 = []
y_test_2013 = []
X_test, y_test = preprocess.load_data('../data_train/twitter2013.test', X_test_2013, y_test_2013)

X_test_2014 = []
y_test_2014 = []
X_test, y_test = preprocess.load_data('../data_train/twitter2014.test', X_test_2014, y_test_2014)

X_test_2015 = []
y_test_2015 = []
X_test, y_test = preprocess.load_data('../data_train/twitter2015.test', X_test_2015, y_test_2015)

X_test_2016 = []
y_test_2016 = []
X_test, y_test = preprocess.load_data('../data_train/twitter2016.test', X_test_2016, y_test_2016)

print "Creating Classifier"
# Create classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=24)
clf3 = SVC(kernel='rbf', class_weight='balanced', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[1,1,2])

print "Training Classifier"

clf1 = clf1.fit(X_train,y_train)
joblib.dump(clf1, '../model/classifier/dtmodel.pkl') 
clf2 = clf2.fit(X_train,y_train)
joblib.dump(clf2, '../model/classifier/knnmodel.pkl') 
clf3 = clf3.fit(X_train,y_train)
joblib.dump(clf3, '../model/classifier/svmmodel.pkl') 
eclf = eclf.fit(X_train,y_train)
joblib.dump(eclf, '../model/classifier/votingmodel.pkl') 

Train classifier and test
print "Twitter 2013 test result: "
train_classifier.train_classifier(clf1, X_test_2013, y_test_2013)
train_classifier.train_classifier(clf2, X_test_2013, y_test_2013)
train_classifier.train_classifier(clf3, X_test_2013, y_test_2013)
train_classifier.train_classifier(eclf, X_test_2013, y_test_2013)

print "Twitter 2014 test result: "
train_classifier.train_classifier(clf1, X_test_2014, y_test_2014)
train_classifier.train_classifier(clf2, X_test_2014, y_test_2014)
train_classifier.train_classifier(clf3, X_test_2014, y_test_2014)
train_classifier.train_classifier(eclf, X_test_2014, y_test_2014)

print "Twitter 2015 test result: "
train_classifier.train_classifier(clf1, X_test_2015, y_test_2015)
train_classifier.train_classifier(clf2, X_test_2015, y_test_2015)
train_classifier.train_classifier(clf3, X_test_2015, y_test_2015)
train_classifier.train_classifier(eclf, X_test_2015, y_test_2015)

print "Twitter 2016 test result: "
train_classifier.train_classifier(clf1, X_test_2016, y_test_2016)
train_classifier.train_classifier(clf2, X_test_2016, y_test_2016)
train_classifier.train_classifier(clf3, X_test_2016, y_test_2016)
train_classifier.train_classifier(eclf, X_test_2016, y_test_2016)