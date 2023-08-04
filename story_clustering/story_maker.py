from __future__ import annotations 
from .eventdetector import Event, extractEventsFromCorpus , sameEvent
from enum import Enum
import time
from .document_representation import Corpus, Document
from datetime import datetime
from .keywords_organizer import KeywordGraph
import math

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

MinKeyGraphCompatibilityEv2St = 0.3
DeltaTimeGap = 0.5

class StoryNode:
    MAX_ID = 1
    def __init__(self, ev:Event, p:StoryNode ) -> None:
        self.id = self.get_ID()
        self.e = ev
        self.parent = p
        if ev is not None:
            self.startTimestamp = ev.getStartTimestamp()
            self.endTimestamp = ev.getEndTimestamp()
        else:
            self.startTimestamp = -1
            self.endTimestamp = -1
        self.numPathNode = 0
        self.consistency = 0.0
        self.TF = dict()
        self.pathTF = dict()
        self.children = list()
    
    def get_ID(self):
        id = StoryNode.MAX_ID 
        StoryNode.MAX_ID += 1
        return id
        
    def setChildren(self,children):
        for child in children:
            child.parent = self
        self.children = children
    
    def numberOfChildren(self):
        return len(self.children)

    def hasChildren(self):
        return (self.numberOfChildren() > 0)
    
    def addChild(self,child: StoryNode):
        child.parent = self
        child.numPathNode = self.numPathNode + 1
        self.children.append(child)
        
    def addChildAt(self,index, child):
        child.parent = self
        child.numPathNode = self.numPathNode + 1
        self.children.insert(index,child)
    
    def isRoot(self):
        return self.parent is None
    
    

class TreeTraversalOrderEnum(Enum):
    PRE_ORDER = 0
    POST_ORDER = 1


class StoryTree:
    MAX_ID = 1
    def __init__(self, ev : Event) -> None:
        self.id = self.get_ID()
        self.root = StoryNode(ev=None,p=None)
        self.root.numPathNode = 0
        self.root.consistency = 0
        self.docTitles = list()
        if ev is not None:
            sn = StoryNode(ev,p=self.root)
            self.root.addChild(sn)
            self.keyGraph = ev.keyGraph
            
            for key in ev.docs.keys():
                self.docTitles.append(ev.docs[key].segTitle)
            self.startTimestamp = ev.getStartTimestamp()
            self.endTimestamp = ev.getEndTimestamp()
        else:
            self.startTimestamp = -1
            self.endTimestamp = -1
        
        self.hotness = 0.0
        self.age = 0
        self.staleAge = 0
        self.summary = ""
        self.graphEdges = dict()
    
    
        
    def get_ID(self):
        id = StoryTree.MAX_ID 
        StoryTree.MAX_ID += 1
        return id
    
    def isEmpty(self):
        return (self.root is None)
    
    def getNumberOfNodes(self):
        numberOfNodes = 0
        if(self.root is not None):
            numberOfNodes = self.auxiliaryGetNumberOfNodes(self.root) + 1
        return numberOfNodes
    
    
    def  getNumberOfNodes(self, node:StoryNode):
        numberOfNodes = 0
        if(node is not None):
            numberOfNodes = self.auxiliaryGetNumberOfNodes(node) + 1
        return numberOfNodes
    
    def auxiliaryGetNumberOfNodes(self,  node:StoryNode):
        numberOfNodes = node.numberOfChildren()
        for child in node.children:
            numberOfNodes += self.auxiliaryGetNumberOfNodes(child)
        return numberOfNodes
    
    def getNumberOfDocs(self):
        return len(self.docTitles)
    
    def getNumberOfDocsByTime(self,timestamp1: int,  timestamp2:int):
        startTimestamp = 0
        endTimestamp = 0
        if (timestamp1 < timestamp2):
            startTimestamp = timestamp1
            endTimestamp = timestamp2
        else:
            startTimestamp = timestamp2
            endTimestamp = timestamp1
        

        numberOfDocs = 0
        storyNodes = self.build(self.root, TreeTraversalOrderEnum.PRE_ORDER)
        for i in range(1,  len(storyNodes)):
            sn = self.storyNodes[i]
            for d in sn.e.docs.values():
                if d.publishTime.getTime() >= startTimestamp and d.publishTime.getTime() <= endTimestamp:
                    numberOfDocs += 1
            
        return numberOfDocs
    
    def getStartTimestamp(self):
        storyNodes = self.build(self.root, TreeTraversalOrderEnum.PRE_ORDER)
        if (len(storyNodes) <= 1):
            return -1
        timestamp = time.time()
        for i in range(1,len(storyNodes)):
            sn = storyNodes[i]
            if (sn.e.getStartTimestamp() < timestamp):
                timestamp = sn.e.getStartTimestamp()
        return timestamp
    
    
    def getEndTimestamp(self):
        storyNodes = self.build(self.root, TreeTraversalOrderEnum.PRE_ORDER)
        if (len(storyNodes) <= 1):
            return -1
        timestamp = -1
        for i in range(1,len(storyNodes)):
            sn = storyNodes[i]
            if (sn.e.getEndTimestamp() > timestamp):
                timestamp = sn.e.getEndTimestamp()
        return timestamp
    
    def build(self,traversalOrder):
        returnList = None
        if(self.root is not None):
            returnList = self.build(self.root, traversalOrder)
        return returnList
    
    def build(self, node : StoryNode, traversalOrder):
        traversalResult = list()

        if (traversalOrder == TreeTraversalOrderEnum.PRE_ORDER):
            self.buildPreOrder(node, traversalResult)
        
        elif (traversalOrder == TreeTraversalOrderEnum.POST_ORDER):
            self.buildPostOrder(node, traversalResult)
        return traversalResult
    
    def buildPreOrder(self, node :StoryNode, traversalResult):
        traversalResult.add(node)

        for child in node.children:
            self.buildPreOrder(child, traversalResult)


    # TODO: to remove is not really used 
    def buildPostOrder(self, node:StoryNode, traversalResult):
        for child in node.children:
            self.buildPostOrder(child, traversalResult)
        traversalResult.add(node)

    
    

    
        
    
    



class StoryForest:
    def __init__(self) -> None:
        self.storyTrees = list()
        self.corpus = Corpus()
        self.cumulativeDF = dict()
        self.cumulativeDocAmount = 0
        
    def getAllEvents(self):
        result = list()
        for st in self.storyTrees:
            storyNodes = st.build(TreeTraversalOrderEnum.PRE_ORDER)
            for i in range(1,len(storyNodes)):
                result.append(storyNodes[i].e)
        return result
    
    def filterStoryTreesByTime(self,t):
        result = list()
        t = datetime.fromtimestamp(t)
        for st in self.storyTrees:
            storyEndTime = datetime.fromtimestamp(st.endTimestamp)
            if (storyEndTime<t):
                result.append(st)
        return result
            
class StoryMaker:
    def __init__(self, corpus: Corpus, model ) -> None:
        self.corpus = corpus
        self.model = model
    
    # generate stories from corpus
    def generateStories(self,events) -> StoryForest:
        
        sf = StoryForest()
        #events = extractEventsFromCorpus(self.corpus)
        sf = self.updateStoriesByEvents(sf, events)
        #sf = self.summarizeStories(sf)
        return sf
    
    # this is done daily
    def updateStoriesByEvents(self, sf: StoryForest, events: list[Event]) -> StoryForest:
        for e in events:
            storyIdx = self.findRelatedStory(e,sf)
            if (storyIdx >= 0):
                self.updateStoryTree(sf,storyIdx, e)
                sf.storyTrees[storyIdx].staleAge = -1
            else:
                newSt = StoryTree(e)
                newSt.staleAge = -1 
                sf.storyTrees.append(newSt)
        
        # we increase the age and stale age of each story tree
        for idx in range(0,len(sf.storyTrees)):
            sf.storyTrees[idx].age += 1
            sf.storyTrees[idx].staleAge += 1
        
        return sf
    
    def updateStoryTree(self, sf : StoryForest, storyIdx : int, e: Event) :
       
        st = sf.storyTrees[storyIdx]
        storyNodes = st.build(TreeTraversalOrderEnum.PRE_ORDER)
        
        maxCompatibility = -1
        matchIdx = -1
        #sameEvent = False
        
        # compare with each story node
        for i in range(1,len(storyNodes)):
            sameEv = self.sameEvent(e, storyNodes[i], sf.corpus.DF, len(sf.corpus.docs))
            if sameEv:
                matchIdx = i
                break
            
            # if not an existing event, calculate compatibility
            compatibility = self.calcCompatibilityEvent2StoryNode(e, storyNodes[i], st)
            if compatibility > maxCompatibility:
                maxCompatibility = compatibility
                matchIdx = i
        
        if sameEv:
            self.merge(e, storyNodes[matchIdx])
        elif maxCompatibility > MinKeyGraphCompatibilityEv2St:
            self.extend(e, storyNodes[matchIdx])
        else:
            self.extend(e, st.root)
            
        # update tree's info
        st.keyGraph = KeywordGraph.mergeKeyGraphs(st.keyGraph, e.keyGraph)
        if (st.startTimestamp > e.getEndTimestamp()):
            st.startTimestamp = e.getStartTimestamp
        if st.endTimestamp < e.getEndTimestamp:
            st.endTimestamp = e.getEndTimestamp()
        
        for d in e.docs.values():
            st.docTitles.append(d.segTitle)
            
            
            
            
            
    
    def findRelatedStory(self,e:Event, sf:StoryForest) -> int:
        matchIdx = -1
        for i in range(0,len(sf.storyTrees)):
            if self.sameStoryByRule(e,sf,i):
                matchIdx = i
                break
        return matchIdx
    
    def sameStoryByRule(self, e:Event, sf:StoryForest, storyTreeIdx:int) -> bool:
        
    
        st = sf.storyTrees[storyTreeIdx]
        
        # check whether there are duplicated document title
        eventDocTitles = set()
        for d in e.docs.values():
            if d.segTitle in st.docTitles: 
                eventDocTitles.add(d.segTitle)
        
        if (len(eventDocTitles) > 0):
            return True
        
        # use sone rules for brand new event
        # compare event and story's keyword graphs
        keyGraphCompatibility = self.calcKeygraphCompatibilityEvent2Story(e,st)
        if keyGraphCompatibility < MinKeyGraphCompatibilityEv2St:
            return False
        
        # Remove stop words
        english_stopwords = stopwords.words('english')
        german_stopwords = stopwords.words('german')
        stopwords = english_stopwords.extend(german_stopwords)
        
        for d in e.docs.values():
            for title in st.docTitles:
                # at least one common keyword
                t = word_tokenize(title.lower())

                tokens_wo_stopwords = [k for k in t if k not in stopwords]
                commonWords = d.titleKeywords.intersect(tokens_wo_stopwords)
                if len(commonWords) > 0:
                    return True
        
        return False
    
    def calcKeygraphCompatibilityEvent2Story(self, e: Event, st:StoryTree) -> float:
        compatibility = 0
        numIntersection = len (set(e.keyGraph.graphNodes.keys()).intersection(set(st.keyGraph.graphNodes.keys())))
        numUnion = len(e.keyGraph.graphNodes.keys())+len(st.keyGraph.graphNodes.keys()) - numIntersection
        if numUnion > 0:
            compatibility = (numIntersection +0.0) / numUnion
        return compatibility
        
    def calcCompatibilityEvent2StoryNode(self, e: Event, sn: StoryNode, st: StoryTree) -> float:
        compatibility = 0
        if e.centroid is None:
            e.calcCentroid()
        if sn.e.centroid is None:
            sn.e.calcCentroid()
        
        # content similarity
        event2StoryNodeCompatibility = Document.cosineSimilarityByTF(e.centroid,sn.e.centroid)
        
        # path similarity
        event2PathCompatibility = (event2StoryNodeCompatibility + (sn.numPathNode -1)*sn.consistency)/sn.numPathNode
        
        # time proximity
        timeProximity = 0
        T = abs(max(st.endTimestamp, e.getEndTimestamp())- min(st.startTimestamp, e.getStartTimestamp()))
        timeGap = 0
        if T != 0:
            timeGap = (e.getStartTimestamp() - sn.e.getStartTimestamp()) / (T + 0.0)
        
        if timeGap >= 0:
            timeProximity = math.exp(-timeGap*DeltaTimeGap)
        else:
            timeProximity = 0 - math.exp(timeGap * DeltaTimeGap)
        
        # calculate comprehensive compatibility
        compatibility = event2StoryNodeCompatibility * event2PathCompatibility * timeProximity

        return compatibility
        
    # TODO: check if correct and if it matches the description in the paper
    def merge(self, e: Event, sn: StoryNode):
        sn.e.docs.update(e.docs)
        sn.startTimestamp = sn.e.getStartTimestamp()
        sn.endTimestamp = sn.e.getEndTimestamp()
    
    def extend(self, e: Event, sn: StoryNode):
        if len(e.docs) > 0:
            newSn = StoryNode(e)
            sn.addChild(newSn)
            if not sn.isRoot():
                event2StoryNodeCompatibility = Document.cosineSimilarityByTF(e.centroid,sn.e.centroid)
                event2PathCompatibility = (event2StoryNodeCompatibility + (sn.numPathNode -1)*sn.consistency)/sn.numPathNode
                newSn.consistency = event2PathCompatibility
            else:
                newSn.consistency = 0
    
    def sameEvent(self, e: Event, sn: StoryNode, DF: dict[str,float], model) -> bool:
        
        # check if duplicated docs
        eDocTitles = set()
        for d in e.docs.values():
            eDocTitles.add(d.segTitle)
            
        snDocTitles = set()
        for d in sn.e.docs.values():
            snDocTitles.add(d.segTitle)
            
        snDocTitles = { t for t in snDocTitles if t in eDocTitles}
        if len(snDocTitles) > 0:
            return True
        
        # get the first document in each document cluster
        keyd1 = next(iter(e.docs))
        keyd2 = next(iter(sn.e.docs))
        
        d1 = e.docs[keyd1]
        d2 = sn.e.docs[keyd2]
        
        return sameEvent(d1,d2,None,None,self.model)
    
    def summarizeStories(self, sf:StoryForest) -> StoryForest:
        # many ways to do this, we use a transformer model to summarize main text representatives 
        # for each story tree
        # TODO: finish this
        pass
        
        
        
        
    
               
    
    
                
                
                
        
        
        
        
        
        
            
            
        
        
        

            
        