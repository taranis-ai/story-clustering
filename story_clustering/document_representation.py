import math
import json
from typing import Any
from .nlp_utils import idf, tfidf


class Keyword:
    """
    A class to represent the keyword data type
    """

    def __init__(self, baseform: str, words: list[str], documents: set  = None, tf: float = 0, df: float = 0):
        self.baseForm = baseform
        self.words = words
        self.documents = documents or set()
        self.tf = tf
        self.df = df

    def increase_tf(self, k):
        self.tf += k

    def increase_df(self, k):
        self.df += k

    def reprJSON(self):
        return {"baseForm": self.baseForm, "words": self.words, "tf": self.tf, "df": self.df, "documents": list(self.documents)}


# class KeywordEncoder(json.JSONEncoder):
#    def default(self, o: Any) -> Any:
#        return {'baseForm': o.baseForm, 'tf': o.tf, 'df': o.df, 'documents': list(o.documents) }


class Document:
    """
    A class to represent a document in a corpus

    tfidfVectorSizeWithKeygraph: double
        Document TF-IDF vector size float consider keygraph keywords. For keywords that doesn't included in a keygraph,
        we don't calculate the keyword's tf.
        This is used to calculate the similarity between a keygraph and a document. It is the norm of document vector,
        where each element in the vector
        is the feature (such as tf, tf-idf) of one document keyword.
    """

    def __init__(
        self,
        doc_id: int = None,
        url: str = None,
        language: str = None,
        title: str = None,
        content: str = None,
        keywords: dict = None,
        publish_date=None,
    ):
        self.doc_id = doc_id
        self.url = url
        self.publish_time = publish_date
        self.language = language
        self.title = title
        if title is not None:
            self.segTitle = title.strip().split(" ")
        self.content = content
        self.keywords = keywords
        self.tf_vector_size: float = -1
        self.tfidf_vector_size: float = -1
        self.tfidfVectorSizeWithKeygraph: float = -1

    def contains_keyword(self, kw):
        return kw in self.keywords

    def set_keyword(self, kw: Keyword):
        if not self.keywords:
            self.keywords = {}
        self.keywords[kw.baseForm] = kw

    # compute document's TF vector size
    # Given document's keywords [w1,..,wn] the vector size of a doc is sqrt(tf(wi)^2)
    def calc_tf_vector_size(self):
        tf_vector_size = 0
        for k in self.keywords.values():
            tf_vector_size += math.pow(k.tf, 2)
        tf_vector_size = math.sqrt(tf_vector_size)
        self.tf_vector_size = tf_vector_size
        return tf_vector_size

    def calc_tfidf_vector_size(self, DF, docSize):
        tfidf_vector_size = 0
        for k in self.keywords.values():
            tfidf_vector_size += math.pow(tfidf(k.tf, idf(DF[k.baseForm], docSize)), 2)
        tfidf_vector_size = math.sqrt(tfidf_vector_size)
        self.tfidf_vector_size = tfidf_vector_size

    @staticmethod
    def cosine_similarity_by_tf(d1, d2) -> float:
        sim = 0
        for k1 in d1.keywords.values():
            if k1.baseForm in d2.keywords:
                tf1 = k1.tf
                tf2 = d2.keywords[k1.baseForm].tf
                sim += tf1 * tf2

        if d1.tf_vector_size < 0:
            d1.calc_tf_vector_size()

        if d2.tf_vector_size < 0:
            d2.calc_tf_vector_size()

        if d1.tf_vector_size == 0 or d2.tf_vector_size == 0:
            return 0

        return sim / d1.tf_vector_size / d2.tf_vector_size

    def reprJSON(self):
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "publish_time": self.publish_time,
            "language": self.language,
            "title": self.title,
            "content": self.content,
            "keywords": [k.reprJSON() for k in self.keywords.values()],
            "tf_vector_size": self.tf_vector_size,
            "tfidf_vector_size": self.tfidf_vector_size,
            "tfidfVectorSizeWithKeygraph": self.tfidfVectorSizeWithKeygraph,
        }


class KeywordEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return {"baseForm": o.baseForm, "tf": o.tf, "df": o.df, "documents": list(o.documents)}


class DocumentEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return {
            "doc_id": o.doc_id,
            "url": o.url,
            "publish_time": o.publish_time,
            "language": o.language,
            "title": o.title,
            "content": o.content,
            "keywords": json.dumps(o.keywords, cls=KeywordEncoder),
            "tf_vector_size": o.tf_vector_size,
            "tfidf_vector_size": o.tfidf_vector_size,
            "tfidfVectorSizeWithKeygraph": o.tfidfVectorSizeWithKeygraph,
        }


class Corpus:
    """
    A class to represent a corpus of documents
    Attributes
    ----------
    docs: dict (doc_id -> Document)
          documents contained in this corpus
    DF: dict (String -> double)
        Words' DF

    Methods:
    ----------
    """

    def __init__(self, docs=None, df=None):
        """
        Args:
            docs (_type_, optional): _description_. Defaults to None.
            df (_type_, optional): _description_. Defaults to None.
        """
        self.docs = {}
        self.DF = {} if df is None else df
        if docs is None:
            return
        for doc_id, doc in enumerate(docs):
            self.docs[doc_id] = doc
            for keyword in doc.keywords().values():
                if keyword.baseForm() in self.DF:
                    self.DF[keyword.baseForm()] += 1
                else:
                    self.DF[keyword.baseForm()] = 1

    def update_df(self):
        self.DF = {}
        for doc in self.docs.values():
            for k in doc.keywords.values():
                self.DF[k.baseForm] = self.DF[k.baseForm] + 1 if k.baseForm in self.DF else 1

    def reprJSON(self):
        return {"docs": self.docs, "DF": self.DF}


class CorpusEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "reprJSON"):
            return o.reprJSON()
        return json.JSONEncoder.default(self, o)

        # json_docs = { k : json.dumps(v,cls=DocumentEncoder) for k,v in o.docs.items()}
        # json_df = json.dumps(o.DF)
