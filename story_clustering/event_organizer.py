import time
from .document_representation import Document, Keyword
from .keywords_organizer import KeywordGraph, KeywordNode, KeywordEdge


class Event:
    def __init__(self):
        self.max_id = 1
        self.keyGraph: KeywordGraph | None = None
        self.docs = {}
        self.similarities = {}
        self.centroid = None
        self.variance = None
        self.summary = None
        self.hotness = None

    # get start timestamp of all docs in this event
    # return -1 if no docs in this event
    def get_start_timestamp(self):
        if len(self.docs) == 0:
            return -1

        timestamp = time.time()
        for doc in self.docs.values():
            if doc.publish_time is not None and doc.publish_time.getTime() < timestamp:
                timestamp = doc.publish_time.getTime()
        return timestamp

    # get end timestamp of all docs in this event
    # return -1 if no doc in this event
    def get_end_timestamp(self):
        if len(self.docs) == 0:
            return -1

        timestamp = -1
        for doc in self.docs.values():
            if doc.publish_time is not None and doc.publish_time.getTime() > timestamp:
                timestamp = doc.publish_time.getTime()
        return timestamp

    # calculate the centroid document of this document cluster
    # centroid is the concatenation of all docs in this event
    def calc_centroid(self):
        self.centroid = Document(-1)
        timestamp = float("inf")
        for doc in self.docs.values():
            if doc.publish_time.getTime() < timestamp:
                timestamp = doc.publish_time.getTime()
            for k in doc.keywords.values():
                if k.baseForm in self.centroid.keywords and self.centroid.keywords:
                    kk = self.centroid.keywords[k.baseForm]
                    kk.tf += k.tf
                    kk.df += k.df
                else:
                    self.centroid.set_keyword(Keyword(k.baseForm, k.word, k.tf, k.df))

        self.centroid.calc_tf_vector_size()
        self.centroid.publish_time = timestamp

    def refine_key_graph(self):
        toRemove = []
        if not self.keyGraph:
            return
        for key in self.keyGraph.graphNodes:
            keywordNode = self.keyGraph.graphNodes[key]
            keyword = keywordNode.keyword.baseForm
            exist = any(d.contains_keyword(keyword) for d in self.docs.values())
            if not exist:
                toRemove.append(keyword)

        for keyword in toRemove:
            self.keyGraph.graphNodes.pop(keyword)
