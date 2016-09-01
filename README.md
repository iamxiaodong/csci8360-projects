# Scalable Document Classification with Naive Bayes in Spark

This is project 1 in CSCI 8360 course at University of Georgia, Spring 2016, we are using the Reuters Corpus, which is a set of news stories split intoa hierarchy of categories. There are multiple class labels per document, our goal is to build a naive bayes model without using any build-in pacakges such as MLLib or scikit-learn. The model achieves 96.8% prediction accuracy in the hold-out data set.


## Getting Started

Here are the instructions to run the python scripts with Spark locally on Mac OS or Linux, if one wants to run this code in Amazon AWS EMR, be sure to change the data path, such as Hadoop path in your master node.

### The data
There are multiple class labels per document, but for the sakeof simplicity we’ll ignore all but the labels ending in CAT: CCAT, GCAT, MCAT and ECAT. There are some documents with more than one CAT label. Treat those documents as ifyou observed the same document once for each CAT label (that is, add to the countersfor all the observed CAT labels).
Here are the available data sets:

```
X_train_vsmall.txt, y_train_vsmall.txt 
X_test_vsmall.txt, y_test_vsmall.txt 
X_train_small.txt, y_train_small.txt 
X_test_small.txt, y_test_small.txt 
X_train_large.txt, y_train_large.txt 
X_test_large.txt
```
### Installing 

Install necessary python modules as below,

```
pip install nltk stop_words numpy  
```

And make sure to download the wordnet corpora in from NLTK site.

```
python -m nltk.downloader wordnet
```
## Procedure
There are three parts in this project: data cleaning/pre-processing, naive bayes modeling, and prediction.  
###Data Cleaning/Pre-processing
We take the following steps to clearn and process the raw data,

* Remove all special characteristics, punctuations, and stopping words. 
* Stem and lemmatize all single words with porter stemmer and wordnet lemmatizer.
* Generate all bigrams, i.e., all pairs of consecutive words, then build the vocabulary with all single words and bigrams.

###Naive Bayes Modeling
* The word and bigram tf-idf scores are calculated with Spark RDD operations.
* Build Laplace smoothing naive bayes model with tf-idf scores.
* Please see [here](https://web.stanford.edu/class/cs124/lec/naivebayes.pdf) for details about naive bayes model with Laplace smoothing.

###Prediction
* For large test data, take the same data cleaning procedures as training set. 
* Predict with the fitted/trained model and write the results in a single text file.
* Submit the text file with predictions to Autolab and get the accuracy results.

## Running
To run the code, be sure change the data path and install all dependent python modules, and run with spark-submit command simply as below,

```
spark-submit main.py
```

## Authors

* **[Xiaodong Jiang](https://www.linkedin.com/in/xiaodongjiang)** - Ph.D. Student, *Department of Statistics*
* **[Yang Song](https://www.linkedin.com/in/yang-song-74298a118/en)** - M.S. Student, *Department of Statistics*
* **Yaotong Cai** - Ph.D. Candidate, *Department of Statistics*
* **Jiankun Zhu** - Ph.D. Student, *Department of Statistics*
   
## Acknowledgments

* Thanks all team members for the laborious work and great collaboration.
* Thanks [Dr. Quinn](http://cobweb.cs.uga.edu/~squinn/), Shang, Shawn, and Usman for the discussions of using super-fancy-and-what-the-hell Amazon AWS, even though we run our final model in local machine for 4 hours.

