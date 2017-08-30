import preprocess

y_test = []
X_test = []

X_test, y_test = preprocess.load_data('../data_train/twitter2013.test', X_test, y_test)
# X_test, y_test = preprocess.load_data('../data_train/twitter2013.test', X_test, y_test)

from sklearn.externals import joblib
dt = joblib.load('../model/classifier/dtmodel.pkl')
dscore = dt.score(X_test, y_test)
print dscore

from sklearn.externals import joblib
knn = joblib.load('../model/classifier/knnmodel.pkl')
kscore = knn.score(X_test, y_test)
print kscore

from sklearn.externals import joblib
svm = joblib.load('../model/classifier/svmmodel.pkl')
sscore = svm.score(X_test, y_test)
print sscore

from sklearn.externals import joblib
vtm = joblib.load('../model/classifier/votingmodel.pkl')
vscore = vtm.score(X_test, y_test)
print vscore