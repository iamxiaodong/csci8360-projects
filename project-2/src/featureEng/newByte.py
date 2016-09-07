from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
from sys import argv
import numpy as np
import re
import urllib2
import csv
from nltk.util import ngrams
import itertools
from collections import Counter

class preprocessor(object):
	
	'''
	preprocess byte data for microsoft malware detection project

	parameters:
	- gramList: a list of integers, representing the number of grams to use 
	  default value: [1, 2], use unigram and bigram
	- freqThreshold: a threshold to filter term (grams) frequency
	  default value: 200

	methods:
	- byteFeatureGenerator(X, y)
	  convert byte file and labels into a sparse matrix
	  parameters:
	    - X: pyspark rdd, with (id, rawDoc) format 
	    - y: pyspark rdd, with (id, label) format  
	''' 
	def __init__(self, grams = [1, 2], freqThreshold = 200): 		
		self.grams = grams
		self.freqThreshold = freqThreshold

	# helper methods	
	def stripFileNames(stringOfName):
    	splits = stringOfName.split(".")
    	name = splits[0][-20:]

    return name

	# token a document, only keep 2-digit code, and its grams
	def tokenEachDoc(aDoc):
		'''
		return a dictionary of item-freq, here items are single words and grams
		'''
		tmpWordList = [x for x in re.sub('\r\n', ' ', aDoc).split() if len(x) == 2]
		tmpGramList = []
		for i in xrange(len(grams)):
			tmpGramList[i] = [''.join(x) for x in list(ngrams(tmpWordList, grams[i]))]
		
		# here tmpGramList is a list of list, here we should remove the inner lists
		sumGramList = tmpWordList + list(itertools.chain.from_iterable(tmpGramList)) # this is a very long list, depends on the gram numbers
		sumGramDict = dict(Counter(sumGramList))
		
		for keys in sumGramDict.keys():
			if sumGramDict[keys] < freqThreshold:
				del sumGramDict[keys]

		return sumGramDict

	def byteFeatureGenerator(X, y): # format, (id, dictionary of items)
		'''
		return an rdd of (id, (freq dictionary of items, label))
		'''
		tokenizedX = X\
		.map(lambda x: (self.stripFileNames(x[0]), self.tokenEachDoc(x[1])))\
		.join(y)

		return tokenizedX

if __name__ == '__main__':

	sc = SparkContext()
	binaryPath = "s3://eds-uga-csci8360/data/project2/binaries/"
	byteFiles = sc.wholeTextFiles(binaryPath) 

	urlTrainX = 'https://s3.amazonaws.com/eds-uga-csci8360/data/project2/labels/' + argv[1]
	trainX = csv.reader(urllib2.urlopen(urlTrainX))

	urlTrainY = 'https://s3.amazonaws.com/eds-uga-csci8360/data/project2/labels/' + argv[2]
	trainY = csv.reader(urllib2.urlopen(urlTrainY))

	trainData = [(x[0][0], x[1][0]) for x in zip(trainX, trainY)]
	labelRdd = sc.parallelize(trainData) # format: (str(id), str(label))

	# feature rdd
	prc = preprocessor()
	byteFeatureRdd = prc.byteFeatureGenerator(byteFiles, labelRdd)

	# save rddd
	print byteFeatureRdd.take(1)










