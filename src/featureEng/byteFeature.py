from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
from sys import argv
import re
import urllib2
import csv
from nltk import bigrams

if len(argv) != 3:
    print """
    Usage: python %s [X_train] [Y_train]
    """ % argv[0]
    exit(1)

sc = SparkContext()

binaryPath = "s3://eds-uga-csci8360/data/project2/binaries/"
byteFiles = sc.wholeTextFiles(binaryPath) 

# doc names are too long, and just keep the last 20 digits, which is its identital name
def stripFileNames(stringOfName):

    splits = stringOfName.split(".")
    name = splits[0][-20:]

    return name

# token a document, only keep 2-digit code, and bigrams
def tokenEachDoc(aDoc):
	
	tmpWordList = [x for x in re.sub('\r\n', ' ', aDoc).split() if len(x) == 2]
	tmpGramList = [''.join(x) for x in list(bigrams(tmpWordList))]
	return tmpWordList + tmpGramList

bytesRdd = byteFiles.map(lambda x: (stripFileNames(x[0]), tokenEachDoc(x[1])))

# ----------
# train sets
# ----------
urlTrainX = 'https://s3.amazonaws.com/eds-uga-csci8360/data/project2/labels/' + argv[1]
trainX = csv.reader(urllib2.urlopen(urlTrainX))

urlTrainY = 'https://s3.amazonaws.com/eds-uga-csci8360/data/project2/labels/' + argv[2]
trainY = csv.reader(urllib2.urlopen(urlTrainY))

trainData = [(x[0][0], x[1][0]) for x in zip(trainX, trainY)]
labelRdd = sc.parallelize(trainData).collect() # format: (str(id), str(label))


# continued: use mllib functions to get a sparse matrix of tfidf-single and tfidf-bigrams
# and save as an RDD














