import math
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
from nltk.tokenize import RegexpTokenizer
from functools import cache
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

POLYFUZZ_THRESHOLD = 3


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
    text_content_tokanized = [replace_umlauts_with_digraphs(w) for w in text_content_tokanized if w.lower() not in get_stop_words()]
    return text_content_tokanized


def compute_tf(baseForm: str, text: str) -> int:
    model = PolyFuzz(TFIDF(model_id="TF-IDF-Sklearn", clean_string=False, n_gram_range=(3, 3)))

    if not baseForm.strip().isalpha() or baseForm in get_stop_words():
        return 1
    n_words = len(baseForm.strip().split(" "))
    tokenized_text = tokanize_text(text)
    try:
        model.match(tokenized_text, [baseForm]).group(link_min_similarity=0.75)
    except Exception:
        return 1
    df = model.get_matches()
    if len(df) == 0:
        # keyword not appearing in text
        return 0
    values = df[(df["Group"].notnull()) & (df["Similarity"] >= 0.65)]  # type: ignore
    if len(values) == 0:
        return 0
    values.loc[:, ("Similarity")] = 1
    tf = max(values["Similarity"].sum(), 1)
    return tf // n_words


def find_keywords_matches(keywords_lst1: list[str], keywords_lst2: list[str]) -> int:
    model = PolyFuzz(TFIDF(model_id="TF-IDF-Sklearn", clean_string=False, n_gram_range=(3, 3)))

    if not keywords_lst1 or not keywords_lst2:
        return 0
    try:
        model.match(keywords_lst1, keywords_lst2).group(link_min_similarity=0.75)
    except Exception:
        return 0
    df = model.get_matches()
    if len(df) == 0:
        return 0
    values = df[(df["Group"].notnull()) & (df["Similarity"] >= 0.65)]  # type: ignore
    values.loc[:, ("Similarity")] = 1
    return max(values["Similarity"].sum(), 1)


def separate_stories(stories: list[dict]) -> tuple[list[dict], list[dict]]:
    already_clustered = []
    to_cluster = []

    for story in stories:
        if len(story["news_items"]) > 1:
            already_clustered.append(story)
        else:
            to_cluster.append(story)

    return already_clustered, to_cluster


@cache
def get_sentence_transformer(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)


@cache
def get_stop_words(languages=None):
    if languages is None:
        languages = ["english", "german"]
    return {word for lang in languages for word in stopwords.words(lang)}
