import time
from .document_representation import Document, Keyword
from .keywords_organizer import KeywordGraph, KeywordNode, KeywordEdge

class Event:
    def __init__(self) -> None:
        self.max_id = 1
        #self.keyGraph = dict()
        self.docs = dict()
        self.similarities = dict()
        self.centroid = None
        self.variance = None
        self.summary = None
        self.hotness = None
        
    # get start timestamp of all docs in this event
    # return -1 if no docs in this event
    def getStartTimestamp(self):
        if len(self.docs) == 0:
            return -1
        
        timestamp = time.time()
        for d in self.docs.values():
            if d.publishTime is not None:
                if d.publishTime.getTime() < timestamp:
                    timestamp = d.publishTime.getTime()
        return timestamp

    # get end timestamp of all docs in this event
    # return -1 if no doc in this event 
    def getEndTimestamp(self):
        if len(self.docs) == 0:
            return -1
        
        timestamp = -1
        for d in self.docs.values():
            if d.publishTime is not None:
                if d.publishTime.getTime()> timestamp:
                    timestamp = d.publishTime.getTime()
        return timestamp
    
    # calculate the centroid document of this document cluster
    # centroid is the concatenation of all docs in this event
    def calcCentroid(self):
        self.centroid = Document('-1')
        timestamp = float('inf')
        for d in self.docs.values():
            if d.publishTime.getTime() < timestamp:
                timestamp = d.publishTime.getTime()
            for k in d.keywords.values():
                if k.baseForm in self.centroid.keywords:
                    kk = self.centroid.keywords[k.baseForm]
                    kk.tf += k.tf
                    kk.df += k.df
                else:
                    self.centroid[k.baseForm] = Keyword(k.baseForm, k.word, k.tf,k.df)
        
        self.centroid.calcTFVectorSize()
        self.centroid.publishTime = timestamp
    
    def refineKeyGraph(self):
        toRemove = list()
        for key in self.keyGraph.graphNodes:
            keywordNode = self.keyGraph.graphNodes[key]
            keyword = keywordNode.keyword.baseForm
            exist = False
            
            for d in self.docs.values():
                if d.containsKeyword(keyword):
                    exist = True
                    break
            
            if not exist:
                toRemove.append(keyword)
        
        for kw in toRemove:
            self.keyGraph.graphNodes.pop(kw)
            #KeywordGraph.removeNode(self.keyGraph,kw)
    