import math
import json
from typing import Any


class Keyword:
    """
    A class to represent the keyword data type
    """

    def __init__(self, baseForm: str, documents: set = set(), tf: float = 0, df: float = 0):
        self.baseForm = baseForm
        self.documents = documents
        self.tf = tf
        self.df = df

    def increase_tf(self, k: float):
        self.tf += k

    def increase_df(self, k: float):
        self.df += k

    def reprJSON(self) -> dict:
        return {"baseForm": self.baseForm, "tf": self.tf, "df": self.df, "documents": list(self.documents)}


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
        doc_id: str | None = None,
        url: str | None = None,
        language: str | None = None,
        title: str | None = None,
        content: str | None = None,
        keywords: dict[str, dict] | None = None,
        publish_time=None,
        tf_vector_size: float = -1,
        tfidf_vector_size: float = -1,
        tfidfVectorSizeWithKeygraph: float = -1,
    ):
        self.doc_id = doc_id
        self.url = url
        self.publish_time = publish_time
        self.language = language
        self.title = title
        if title is not None:
            self.segTitle = title.strip().split(" ")
        self.content = content
        if keywords:
            self.keywords = {k: Keyword(**v) for k, v in keywords.items()}  # type: ignore
        self.tf_vector_size = tf_vector_size
        self.tfidf_vector_size = tfidf_vector_size
        self.tfidfVectorSizeWithKeygraph = tfidfVectorSizeWithKeygraph

    def contains_keyword(self, kw: str) -> bool:
        return kw in self.keywords

    def set_keyword(self, kw: Keyword):
        if not self.keywords:
            self.keywords = {}
        self.keywords[kw.baseForm] = kw

    def calc_tf_vector_size(self) -> float:
        # compute documents term frequency (tf) vector size
        # Given documents keywords [kw_1,..,kw_n] the vector size of the tf is sqrt[sum_i( tf(kw_i)^2 )]
        tf_vector_size = sum(math.pow(k.tf, 2) for k in self.keywords.values())
        tf_vector_size = math.sqrt(tf_vector_size)
        self.tf_vector_size = tf_vector_size
        return tf_vector_size

    @staticmethod
    def cosine_similarity_by_tf(d1: "Document", d2: "Document") -> float:
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

    def reprJSON(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "publish_time": self.publish_time,
            "language": self.language,
            "title": self.title,
            "content": self.content,
            "keywords": {k: v.reprJSON() for k, v in self.keywords.items()},
            "tf_vector_size": self.tf_vector_size,
            "tfidf_vector_size": self.tfidf_vector_size,
            "tfidfVectorSizeWithKeygraph": self.tfidfVectorSizeWithKeygraph,
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

    """

    def __init__(self, docs: dict[int, dict] | None = None, DF=None):
        self.DF = {} if DF is None else DF
        self.docs = {}
        if docs:
            for doc in docs.values():
                self.docs[doc["doc_id"]] = Document(**doc)
                for key, keyword in self.docs[doc["doc_id"]].keywords.items():
                    if keyword.baseForm in self.DF:
                        self.DF[keyword.baseForm] += 1
                    else:
                        self.DF[keyword.baseForm] = 1

    def update_df(self):
        self.DF = {}
        for doc in self.docs.values():
            for k in doc.keywords.values():
                self.DF[k.baseForm] = self.DF[k.baseForm] + 1 if k.baseForm in self.DF else 1

    def reprJSON(self):
        return {"docs": {doc_id: doc.reprJSON() for doc_id, doc in self.docs.items()}, "DF": self.DF}


class CorpusEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "reprJSON"):
            return o.reprJSON()
        return json.JSONEncoder.default(self, o)

        # json_docs = { k : json.dumps(v,cls=DocumentEncoder) for k,v in o.docs.items()}
        # json_df = json.dumps(o.DF)
