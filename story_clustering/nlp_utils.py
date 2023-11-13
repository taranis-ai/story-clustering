import math
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
from nltk.tokenize import RegexpTokenizer
from story_clustering import stopwords_list


# calculate word's inverse document frequency
def idf(df: float, size: int):
    return math.log(size / (df + 1)) / math.log(2)


def replace_umlauts_with_digraphs(s: str) -> str:
    return s.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")

# calculate word's tf-idf
def tfidf(tf, idf):
    return 0 if tf == 0 or idf == 0 else tf * idf


def tokanize_text(text: str) -> list[str]:
    tokenizer = RegexpTokenizer(r"\w+")
    text_content_tokanized = tokenizer.tokenize(text)
    text_content_tokanized = [replace_umlauts_with_digraphs(w) for w in text_content_tokanized if w.lower() not in stopwords_list]
    return text_content_tokanized


def compute_tf(baseForm, text):
    model = PolyFuzz(TFIDF(model_id="TF-IDF-Sklearn", clean_string=False, n_gram_range=(3, 3)))

    if not baseForm.strip().isalpha() or baseForm in stopwords_list:
        return 1
    n_words = len(baseForm.strip().split(" "))
    tokenized_text = tokanize_text(text)
    try:
        model.match(tokenized_text, [baseForm]).group(link_min_similarity=0.75)
    except:
        return 0
    df = model.get_matches()
    if len(df) == 0:
        # keyword not appearing in text
        return 0
    values = df[(df["Group"].notnull()) & (df["Similarity"] >= 0.65)]
    if len(values) == 0:
        return 0
    values.loc[:, ("Similarity")] = 1
    tf = max(values["Similarity"].sum(), 1)
    return tf / n_words
