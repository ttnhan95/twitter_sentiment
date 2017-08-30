import csv
import re
from nltk import ngrams
import preprocess

def get_data_tsv(filename, i):
    content = ""
    f = open(filename, 'r')
    contents = f.read()
    for row in contents.split("\n"):
        tweet = row.split("\t")[i]
        if tweet != "Not Available":
            processedTweet = preprocess.processTweet(tweet)
            processedTweet = preprocess.emoConvert(processedTweet)
            processedTweet = preprocess.aggreConvert(processedTweet)
            content += processedTweet.strip() + "\n"
    return content

def get_data_csv():
    content = ""
    with open('../data/data.csv', 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            sentiment = row[1]
            tweet = row[3]
            if tweet != "Not Available":
                processedTweet = preprocess.processTweet(tweet)
                processedTweet = preprocess.emoConvert(processedTweet)
                processedTweet = preprocess.aggreConvert(processedTweet)
                content += processedTweet.strip() + "\n"
    return content