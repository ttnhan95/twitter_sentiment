#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import csv
import re
import numpy as np
from sklearn import svm
import fasttext
from senti_classifier import senti_classifier

negations = ["no", "not", "none", "no one", "nobody", "nothing", "neither", "nowhere", "never", "hardly", "scarely", "barely", "doesn’t", "isn’t", "wasn’t", "shouldn’t", "wouldn’t", "couldn’t", "won’t", "can’t", "don’t"]

def checkASCII(sentence):
	for c in sentence:
		if ord(c)>128:
			return False
	return True

def checkNegation(tweet):
	for word in tweet.split():
		if word.lower() in negations:
			return 1
	return 0

def processTweet(tweet):
	# process the tweets

	#Convert to lower case
	tweet = tweet.lower()
	#Convert www.* or https?://* to URL
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	#Convert @username to AT_USER
	tweet = re.sub('@[^\s]+','AT_USER',tweet)
	#Remove additional white spaces
	tweet = re.sub('[\s]+', ' ', tweet)
	#Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	#replace 2 or more
	tweet = re.sub(r'([a-z])\1{2,}', r'\1\1', tweet)
	#trim
	tweet = tweet.strip('\'"')
	return tweet
#end

def emoConvert(tweet):
	f = open("../data/emo.txt", 'r')
	emo_contents = f.read()
	for row in emo_contents.split("\n"):
		word = row.split("\t")[1]
		replace = row.split("\t")[0]
		# k = re.compile(r'\b%s\b' % re.escape(word))
		# tweet = k.sub(replace, tweet)
		tweet = tweet.replace(word, replace)
	return tweet

def aggreConvert(tweet):
	f = open("../data/aggregation.txt", 'r')
	agg_contents = f.read()
	for row in agg_contents.split("\n"):
		word = row.split(":")[0]
		replace = row.split(":")[1]
		# k = re.compile(r'\b%s\b' % re.escape(word))
		# tweet = k.sub(replace, tweet)
		tweet = tweet.replace(" " + word + " ", " " + replace + " ")
	return tweet

def get_sentiment(i):
	if i == "positive":
		return 2
	elif i == "negative":
		return 0
	else:
		return 1

# Loading some example data
def load_data(filename, x, y):
	f = open(filename, 'r')
	file_contents = f.read()
	f.close()
	lines = file_contents.split('\n')
	for l in lines:
		if l != "":
			y.append(int(l.split('\t')[0]))
			feature = l.split('\t')[1].split()
			arr = np.array(feature)
			# np.delete(arr, 302)
			final = map(float, arr)
			x.append(final)

	return x,y