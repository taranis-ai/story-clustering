import math
from nltk.corpus import stopwords
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


stopwords_list = stopwords.words('english')
stopwords_list.extend(stopwords.words('german'))

# calculate word's inverse document frequency
def idf(df: float, size: int):
    return math.log(size / (df + 1)) / math.log(2)


# calculate word's tf-idf
def tfidf(tf, idf):
    return 0 if tf == 0 or idf == 0 else tf * idf


def tokanize_text(text:str) -> list[str]:
    tokenizer = RegexpTokenizer(r'\w+')
    text_content_tokanized = tokenizer.tokenize(text)
    text_content_tokanized = [w.lower() for w in text_content_tokanized if w.lower() not in stopwords_list]
    return text_content_tokanized

def compute_tf(baseForm, words, text):
    tfidf = TFIDF(model_id="TF-IDF-Sklearn",clean_string=False,n_gram_range=(3, 3))
    model = PolyFuzz(tfidf)
     
    if not baseForm.strip().isalpha() or len(baseForm.strip()) <=2 or baseForm in stopwords_list:
        return 1
    n_words = len(baseForm.strip().split(" "))  
    all_words = list(words)
    all_words.append(baseForm)
    model.match(tokanize_text(text),all_words).group(link_min_similarity=0.75)
    df = model.get_matches()
    
    values = df[(df["Group"].notnull()) & (df["Similarity"] >=0.65) ]
    if len(values) == 0:
        return 0
    values.loc[:,("Similarity")] = 1
    tf = values["Similarity"].sum()
    if (tf < 1):
        tf = 1
    return tf/n_words