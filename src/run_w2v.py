#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import csv
import re
import numpy as np
from sklearn import svm
import fasttext
from senti_classifier import senti_classifier
import preprocess
import os
import os.path

#load model w2v
model = fasttext.load_model('../model/twittermodel.bin')

negations = ["not", "never", "hardly", "scarely", "barely", "doesn’t", "isn’t", "wasn’t", "shouldn’t", "wouldn’t", "couldn’t", "won’t", "can’t", "don’t"]
# negation_special = ["no", "none", "no one", "nobody", "nothing", "neither", "nowhere", ]
negation_special = ["no one", "nobody", "nothing", "neither", "nowhere", ]

contrast_word = ["but", "despite", "although", "though", "however"]

def find_contrast(sentence):
	for word in re.split(r'[ ,:"]',row):
		if word.lower() in contrast_word:
			return word
		return ""

#start getfeatureVector
def getFeatureVector(tweet):
	sum_vector = np.zeros(300)
	words = tweet.split()
	# for word in words:
	# 	sum_vector += np.array(model[word])
	for i, word in enumerate(words):
		if i > 0:
			if words[0] in negation_special:
				sum_vector -= np.array(model[word])
			# elif word[i-1] in negation:
			# 	sum_vector -= np.array(model[word])
			else:
				sum_vector += np.array(model[word])
	sentences = []
	sentences.append(tweet)
	pos_score, neg_score = senti_classifier.polarity_scores(sentences)
	negation = preprocess.checkNegation(tweet)
	sum_vector = np.concatenate((sum_vector, [pos_score, neg_score, negation]), axis=0)
	return sum_vector

#get feature vector of a data file

def get_w2v(filename, output):
	print "Getiing Feature Vector: " + filename
	if os.path.isfile(output):
		os.remove(output)
	f = open(filename, 'r')
	rows = f.read().split("\n")
	for i, row in enumerate(rows):
		content=""
		if row!="":
			sentiment = preprocess.get_sentiment(row.split("\t")[1])
			if len(row.split("\t")) > 1:
				tweet = row.split("\t")[2]
				if "." in tweet or "?" in tweet or "!" in tweet:
					tweet=""
					sentences = re.split(r'[.?!]',row)
					for sentence in sentences:
						if find_contrast(sentence)!="":
							if sentence.split()[0].lower() in contrast_word:
								if "," in tweet:
									i = sentence.index(",") + 1 - len(sentence)
									sentence = sentence[i:]
							else:
								i = sentence.index(find_contrast(sentence)) + 1 - len(sentence) - len(find_contrast(sentence))
								sentence = sentence[i:]
					tweet += sentence + "."
				else:
					if find_contrast(tweet)!="":
						if tweet.split()[0].lower() in contrast_word:
							if "," in tweet:
								i = tweet.index(",") + 1 - len(tweet)
								tweet = tweet[i:]
						else:
							i = tweet.index(find_contrast(tweet)) + 1 - len(tweet) - len(find_contrast(tweet))
							tweet = tweet[i:]

				if tweet != "Not Available":
					processedTweet = preprocess.processTweet(tweet)
					processedTweet = preprocess.emoConvert(processedTweet)
					processedTweet = preprocess.aggreConvert(processedTweet)
					vector = getFeatureVector(processedTweet.strip().decode('utf8'))
					for f in vector:
						content += str(f) + " "
					with open(output, "a") as text_file:
						text_file.write(str(sentiment) + "\t" + content + "\n")

# def get_w2v(filename, output):
# 	print "Getiing Feature Vector: " + filename
# 	if os.path.isfile(output):
# 		os.remove(output)
# 	f = open(filename, 'r')
# 	rows = f.read().split("\n")
# 	for i, row in enumerate(rows):
# 		content=""
# 		if row!="":
# 			sentiment = preprocess.get_sentiment(row.split("\t")[1])
# 			if len(row.split("\t")) > 1:
# 				tweet = row.split("\t")[2]
# 				if tweet != "Not Available":
# 					processedTweet = preprocess.processTweet(tweet)
# 					processedTweet = preprocess.emoConvert(processedTweet)
# 					processedTweet = preprocess.aggreConvert(processedTweet)
# 					vector = getFeatureVector(processedTweet.strip().decode('utf8'))
# 					for f in vector:
# 						content += str(f) + " "
# 					with open(output, "a") as text_file:
# 						text_file.write(str(sentiment) + "\t" + content + "\n")