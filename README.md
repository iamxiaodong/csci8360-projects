##Scalable Malware Classification 

###pyspark Functions

####preprocessor
Inuput: 500 GB raw byte files

Output: Sparse matrix, where columns are two-digit hexadecimal tf-idf and its bigrams/trigram' tf-idf scores, rows are its identital id.

* Put all byte files into a sinlge paired RDD, (id, doc), and parse each doc into single words and bigrams/trigrams (NLTK bigram/trigram functions), then calculate their tf-idf scores, with MLlib, see **[here](https://spark.apache.org/docs/latest/ml-features.html)**.  
* 