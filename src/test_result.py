#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import csv
import re
import numpy as np
from sklearn import svm
import fasttext
from senti_classifier import senti_classifier
import preprocess

#load model w2v
model = fasttext.load_model('../model/twittermodel.bin')

#start getfeatureVector
def getFeatureVector(tweet):
	sum_vector = np.zeros(300)
	words = tweet.split()
	for word in words:
		sum_vector += np.array(model[word])
	sentences = []
	sentences.append(tweet)
	pos_score, neg_score = senti_classifier.polarity_scores(sentences)
	negation = preprocess.checkNegation(tweet)
	sum_vector = np.concatenate((sum_vector, [pos_score, neg_score, negation]), axis=0)
	return sum_vector

#load model classifier
from sklearn.externals import joblib
clf = joblib.load('../model/classifier/votingmodel.pkl')
print "model done"

while True:
	x = raw_input('Enter a comment: ')
	x = preprocess.processTweet(x)
	x = preprocess.emoConvert(x)
	x = preprocess.aggreConvert(x)
	V = getFeatureVector(x)

	result = clf.predict(V)
	print result