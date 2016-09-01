from __future__ import print_function
import string
import re
from operator import add
from pyspark import SparkContext
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
from nltk import bigrams


sc = SparkContext()
# -------------------
# Data Pre-processing
# -------------------
# read raw training set
xData = sc.textFile('/Users/xiaodong/Documents/course/CSCI8360/project-1/X_train_large.txt')
numOfDocs = xData.count()

# define the stop words list, containing ''
stop_words = get_stop_words('english')
stopWordsList = [re.sub("'", '', x) for x in stop_words] + ['']

# stemmer and lemmatizer from NLTK module
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def convertWords(wordStr, idx):
    '''
    Here we may need to add the grams options in this convert print_function
    first, add bigrams to the word list
    '''
    tmpWordList = re.sub('[^a-zA-Z ]', '', wordStr).split()
    newWordList = [porter_stemmer.stem(wordnet_lemmatizer.lemmatize(x.lower())) for x in tmpWordList if x.lower() not in stopWordsList]
    tmpGrams = [''.join(x) for x in list(bigrams(newWordList))]
    tmp =  tmpGrams + newWordList

    wordDocPairs = []
    for i in xrange(len(tmp)):
        wordDocPairs.append(str(idx) + '-' + tmp[i])
    return wordDocPairs

def tfIdf(word, tup):
    '''
    Here we apply the tf-idf transformation to the words.
    '''
    docID = tup[0].split('-')[0]
    wordFre = tup[0].split('-')[1]
    a = str(docID) + '-' + word
    b = float(wordFre) * float(tup[1])
    return (a, b)

xLine = xData.zipWithIndex()\
    .map(lambda x: convertWords(x[0], x[1]))\
    .flatMap(lambda x: x)\
    .map(lambda x: (x, 1))\
    .reduceByKey(lambda x, y: x + y)

wordDf = xLine.map(lambda x: (x[0].split('-')[1], 1))\
    .reduceByKey(lambda x, y: x + y)

wordIdf = wordDf.map(lambda x: (x[0], np.log((float(numOfDocs)+1)/(float(x[1])+1))))

wordTfIdf = xLine.map(lambda x: (x[0].split('-')[1], str(x[0].split('-')[0]) + '-'+ str(x[1])))\
    .join(wordIdf)\
    .map(lambda x: tfIdf(x[0], x[1]))\
    .map(lambda x: (x[0].split('-')[0], (x[0].split('-')[1], x[1])))
    #.map(lambda x: (x[0], (x[1][0], x[1][1] ** 0.5)))

# ----------------------------------
#  Response variable in training set
# ----------------------------------
yData = sc.textFile('/Users/xiaodong/Documents/course/CSCI8360/project-1/y_train_large.txt')

def catIntersect(line):
    '''
    Preprocess the labels of Docs.
    '''
    cat = list(set(line.split(",")).intersection(set(['ECAT', 'CCAT', 'GCAT', 'MCAT'])))
    return [str(x) for x in cat]

yLine = yData.zipWithIndex()\
    .map(lambda x: (str(x[1]), catIntersect(x[0])))


res = wordTfIdf.join(yLine)\
    .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1])))\
    .sortByKey()\
    .filter(lambda x: x[1][2] != [])

# --------------------------------------------
# Naive Bayes training with Laplace smoothing
# --------------------------------------------
def enlarge(aWord):
    return (('ECAT-' + aWord, 'tmp'), ('MCAT-' + aWord, 'tmp'), ('CCAT-' + aWord, 'tmp'), ('GCAT-' + aWord, 'tmp'))

vocabularyList = wordTfIdf.map(lambda x: x[1][0]).distinct().flatMap(lambda x: enlarge(x))
lengthOfVocabulary = wordTfIdf.map(lambda x: x[1][0]).distinct().count()

def flatMapLabels(labelList):
    tmp = []
    for i in xrange(len(labelList)):
        tmp.append((labelList[i], 1))
    return tmp

categoryNum = yLine.map(lambda x: flatMapLabels(x[1])).flatMap(lambda x: x).reduceByKey(lambda x, y: x+y)
labelNumDict = dict(categoryNum.collect())
sumLabel = sum(labelNumDict.values())
labelRatioDict = dict(categoryNum.map(lambda x: (x[0], np.log(float(x[1])/float(sumLabel)))).collect())

# new dictionary
def splitFun(aList):
    newList = []
    if len(aList) == 1:
        return [(aList[0], 1)]
    else:
        N = len(aList)
        for i in xrange(N):
            newList.append((aList[i], 1))
        return newList

wordNumInClass = res.flatMap(lambda x: splitFun(x[1][2])).reduceByKey(add)
wordNumDict = dict(wordNumInClass.collect())

def combineStrList(StrValue, catList, tfidf):
    tmp = []
    for i in xrange(len(catList)):
        tmp.append((catList[i] + '-' + StrValue, tfidf))
    return tmp

# Format: ('class-word', tfidfsum)
tfIdfSum = res.flatMap(lambda x: combineStrList(x[1][0], x[1][2], x[1][1]))\
    .reduceByKey(lambda x, y: x+y)

# Build a dictionary of missing values, i.e., missing words in certain classes
missingDict = {'MCAT': 1.0/(wordNumDict['MCAT'] + lengthOfVocabulary),
               'ECAT': 1.0/(wordNumDict['ECAT'] + lengthOfVocabulary),
               'GCAT': 1.0/(wordNumDict['GCAT'] + lengthOfVocabulary),
               'CCAT': 1.0/(wordNumDict['CCAT'] + lengthOfVocabulary)}

# Impute missing values
def missingImpute(aTuple):
    # ('class-word', (tfidf, 'tmp'))
    if aTuple[1][0] == None:
        return (aTuple[0], missingDict[aTuple[0].split('-')[0]])
    else:
        return (aTuple[0], aTuple[1][0])

fullTfidfSum = tfIdfSum.fullOuterJoin(vocabularyList).map(lambda x: missingImpute(x))

# Calculate P_hat(w|c)
# Format: (word, (class, p(w|label)))
def wordClassProb(tfIdfSummation, wordNumbyClass, lengthofVoc):
    return np.log((tfIdfSummation + 1.0)/float(wordNumbyClass + lengthofVoc))

postProb = fullTfidfSum\
    .map(lambda x: (x[0].split('-')[1], (x[0].split('-')[0], wordClassProb(x[1], wordNumDict[x[0].split('-')[0]], lengthOfVocabulary))))

# ------------
# Testing
# ------------
xDataTest = sc.textFile('/Users/xiaodong/Documents/course/CSCI8360/project-1/X_test_large.txt')

def getLabel(aStringList):
    tmp = [x for x in aStringList.split('-') if x != '']
    floatlist = [float(tmp[i]) for i in [1,3,5,7]]
    largestIndex = floatlist.index(min(floatlist))
    return tmp[2*largestIndex]

# Format: (id, (word, class, p(w|c)))
xLineTest = xDataTest.zipWithIndex()\
    .map(lambda x: convertWords(x[0], x[1]))\
    .flatMap(lambda x: x)\
    .map(lambda x: (x.split('-')[1], x.split('-')[0]))\
    .join(postProb)\
    .map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1])))\
    .map(lambda x: (x[0] + '-' + x[1][1], x[1][2]))\
    .reduceByKey(add)\
    .map(lambda x: (x[0], float(x[1]) + float(labelRatioDict[x[0].split('-')[1]])))\
    .map(lambda x: (x[0].split('-')[0], ('-' + x[0].split('-')[1] + '-' + str(x[1]))))\
    .reduceByKey(add)\
    .map(lambda x:(float(x[0]), getLabel(x[1])))\
    .sortByKey()


# --------------------------------------------
# Output the prediction results
# --------------------------------------------
predictTest = xLineTest.collect()
f = open('predictTestLargeBi_1-0', 'w')
for i in xrange(len(predictTest)):
    f.write(str(predictTest[i][1])+'\n')

# --------------------------------------------
# Print the precision in testing set
# --------------------------------------------

# yDataTest = sc.textFile('/Users/xiaodong/Documents/course/CSCI8360/project1/y_test_small.txt').collect()
#
# x = [x[1] for x in predictTest]
#
# tmp = []
# for i in xrange(len(x)):
#     if str(x[i]) in yDataTest[i].split(','):
#         tmp.append(1)
#     else:
#         tmp.append(0)
#
# #print(tmp)
# print(float(sum(tmp))/float(len(tmp)))

sc.stop()

