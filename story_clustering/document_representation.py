from collections.abc import Callable
import math
from typing import Any
from .nlp_utils import idf,tfidf
from typing_extensions import Self
import json
class Keyword:
    """
    A class to represent the keyword data type
    Attributes
    -------------
    baseForm: str
        Base form of the word
    word: list[str]
        List of words with similar form
    tf: double
        Term frequency of the word 
    df: double
        Document frequency of the word
    documents: set
        Set of document ids containing this word
    Methods
    ------------
    TODO
    """
    def __init__(self,baseform, words, documents=None, tf=0, df=0) -> None:
        self.baseForm = baseform
        self.words = words
        self.tf = tf
        self.df = df
        if documents != None:
            self.documents = documents
        else:
            self.documents = set()
    
    def increase_tf(self,k) -> None:
        self.tf += k
    
    def increase_df(self,k) -> None:
        self.df += k
    
    def documents(self) -> set():
        return self.documents
    
    def tf(self):
        return self.tf
    
    def df(self):
        return self.df
    
    def reprJSON(self):
        return {'baseForm': self.baseForm, 'words':self.words, 'tf': self.tf, 'df': self.df, 'documents': list(self.documents) }
    #def add_document(self, doc_id):
        
        
#class KeywordEncoder(json.JSONEncoder):
#    def default(self, o: Any) -> Any:
#        return {'baseForm': o.baseForm, 'tf': o.tf, 'df': o.df, 'documents': list(o.documents) }
        

    
class Document:
    """
    A class to represent a document in a corpus
    Attributes
    ------------
    doc_id: int
           document id
    url: str
         document URL
    title: str
         document title
    topic: str
        document topic category
    publishTime: date
        document publish date
    language: str
        document language
    keywords: dict (id-> str)
        map (keywords id, keyword) for content keywords
    content: str
        Document content
    tfitfVecorSizeWithKeygraph: double
        Document TF-IDF vector size that consider keygraph keywords. For keywords that doesn't included in a keygraph, we don't calculate the keyword's tf. 
        This is used to calculate the similarity between a keygraph and a document. It is the norm of document vector, where each element in the vector 
        is the feature (such as tf, tf-idf) of one document keyword.
    tfidfVectorSize: double
        Document TF-IDF vector size.
    tfVectorSize: double
        Document TF vector's size.
    processed: bool
        Whether this document has been processed.
        
    Methods
    -----------
    TODO
    
    """
    def __init__(self, doc_id, url = None, publish_date = None, language = None, title = None, content = None, keywords = None):
        self.doc_id = doc_id
        self.url = url
        self.publishTime = publish_date
        self.language = language
        self.title = title
        self.segTitle = title.strip().split(' ')
        self.content = content
        self.keywords = keywords
        self.tfVectorSize = -1
        self.tfidfVectorSize = -1
        self.tfidfVectorSizeWithKeygraph = -1
    
    def containsKeyword(self,kw):
        if kw in self.keywords:
            return True
        return False
    # compute document's TF vector size
    # Given document's keywords [w1,..,wn] the vector size of a doc is sqrt(tf(wi)^2)
    def calcTFVectorSize(self):
        tfVectorSize = 0
        for k in self.keywords.values():
            tfVectorSize += math.pow(k.tf,2)
        
        tfVectorSize = math.sqrt(tfVectorSize)
        self.tfVectorSize = tfVectorSize
        return tfVectorSize
    
    def calcTFIDFVectorSize(self, DF, docSize):
        tfidfVectorSize = 0
        for k in self.keywords.values():
            tfidfVectorSize += math.pow(tfidf(k.tf, idf(DF[k.baseForm], docSize)),2)
        tfidfVectorSize = math.sqrt(tfidfVectorSize)
        self.tfidfVectorSize = tfidfVectorSize
        return 
    
    @staticmethod
    def cosineSimilarityByTF(d1: Self, d2 : Self) -> float:
        sim = 0
        for k1 in d1.keywords.values():
            if k1.baseForm in d2.keywords:
                tf1 = k1.tf
                tf2 = d2.keywords[k1.baseForm].tf
                sim += tf1 * tf2
        
        if d1.tfVectorSize < 0:
            d1.calcTFVectorSize()
        
        if d2.tfVectorSize < 0:
            d2.calcTFVectorSize()
        
        if d1.tfVectorSize == 0 or d2.tfVectorSize == 0:
            return 0
        else:
            return sim / d1.tfVectorSize / d2.tfVectorSize
    
    def reprJSON(self):
        return {'doc_id': self.doc_id, 'url': self.url, 'publish_time':self.publishTime, 'language':self.language,
                'title': self.title, 'content': self.content, 'keywords': [k.reprJSON() for k in self.keywords.values()], 'tfVectorSize': self.tfVectorSize,
                'tfidfVectorSize': self.tfidfVectorSize, 'tfidfVectorSizeWithKeygraph': self.tfidfVectorSizeWithKeygraph}
        
class DocumentEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return {'doc_id': o.doc_id, 'url': o.url, 'publish_time':o.publishTime, 'language':o.language,
                'title': o.title, 'content': o.content, 'keywords': json.dumps(o.keywords, cls=KeywordEncoder), 'tfVectorSize': o.tfVectorSize,
                'tfidfVectorSize': o.tfidfVectorSize, 'tfidfVectorSizeWithKeygraph': o.tfidfVectorSizeWithKeygraph}
    

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
    def __init__(self, docs = None, df = None) -> None:
        """
        Args:
            docs (_type_, optional): _description_. Defaults to None.
            df (_type_, optional): _description_. Defaults to None.
        """        
        self.docs = dict()
        if df is None:
            self.DF = dict()
        else:
            self.DF = df
        if docs is not None:
            for doc_id,doc in enumerate(docs):
                self.docs[doc_id] = doc
                for keyword in doc.keywords().values():
                    if keyword.baseForm() in self.DF:
                        self.DF[keyword.baseForm()] +=1
                    else:
                        self.DF[keyword.baseForm()] = 1
    

    
    def updateDF(self):
        self.DF = dict()
        for d in self.docs.values():
            for k in d.keywords.values():
                if k.baseForm in self.DF:
                    self.DF[k.baseForm] = self.DF[k.baseForm] + 1
                else:
                    self.DF[k.baseForm] = 1
                    
    def reprJSON(self):
        return {'docs': self.docs, 'DF':self.DF}
        
                     
class CorpusEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o,'reprJSON'):
            return o.reprJSON()
        else:
            return json.JSONEncoder.default(self, o)


        
        
        
        #json_docs = { k : json.dumps(v,cls=DocumentEncoder) for k,v in o.docs.items()}
        #json_df = json.dumps(o.DF)
        
        
        