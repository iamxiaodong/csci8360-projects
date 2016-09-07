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

asmPath = "s3://eds-uga-csci8360/data/project2/metadata/"
asmFiles = sc.wholeTextFiles(asmPath) 

# ----------
# train sets
# ----------
urlTrainX = 'https://s3.amazonaws.com/eds-uga-csci8360/data/project2/labels/' + argv[1]
trainX = csv.reader(urllib2.urlopen(urlTrainX))

urlTrainY = 'https://s3.amazonaws.com/eds-uga-csci8360/data/project2/labels/' + argv[2]
trainY = csv.reader(urllib2.urlopen(urlTrainY))

trainData = [(x[0][0], x[1][0]) for x in zip(trainX, trainY)]
labelRdd = sc.parallelize(trainData).collect() # format: (str(id), str(label))
