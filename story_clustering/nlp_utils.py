import math
from nltk.corpus import stopwords



# calculate word's inverse document frequency
def idf(df: float,size: int):
    return math.log(size/(df+1)) / math.log(2)


# calculate word's tf-idf
def tfidf(tf,idf):
    if tf == 0 or idf == 0:
        return 0
    return tf*idf

