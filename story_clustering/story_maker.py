from __future__ import annotations
from datetime import datetime
from enum import Enum
import time
import math

from .eventdetector import Event, same_event
from .document_representation import Corpus, Document
from .keywords_organizer import KeywordGraph


import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

MinKeyGraphCompatibilityEv2St = 0.3
DeltaTimeGap = 0.5


class StoryNode:
    MAX_ID = 1

    def __init__(self, ev: Event | None = None, p: StoryNode | None = None):
        self.id = self.get_ID()
        self.e = ev
        self.parent = p
        if ev is not None:
            self.startTimestamp = ev.get_start_timestamp()
            self.endTimestamp = ev.get_end_timestamp()
        else:
            self.startTimestamp = -1
            self.endTimestamp = -1
        self.numPathNode = 0
        self.consistency = 0.0
        self.TF = {}
        self.pathTF = {}
        self.children = []

    def get_ID(self):
        max_id = StoryNode.MAX_ID
        StoryNode.MAX_ID += 1
        return max_id

    def set_children(self, children):
        for child in children:
            child.parent = self
        self.children = children

    def number_of_children(self):
        return len(self.children)

    def has_children(self):
        return self.number_of_children() > 0

    def addChild(self, child: StoryNode):
        child.parent = self
        child.numPathNode = self.numPathNode + 1
        self.children.append(child)

    def add_child_at(self, index, child):
        child.parent = self
        child.numPathNode = self.numPathNode + 1
        self.children.insert(index, child)

    def isRoot(self):
        return self.parent is None


class TreeTraversalOrderEnum(Enum):
    PRE_ORDER = 0
    POST_ORDER = 1


class StoryTree:
    MAX_ID = 1

    def __init__(self, ev: Event):
        self.id = self.get_ID()
        self.root = StoryNode(ev=None, p=None)
        self.root.numPathNode = 0
        self.root.consistency = 0
        self.docTitles = []
        if ev is not None:
            sn = StoryNode(ev, p=self.root)
            self.root.addChild(sn)
            self.keyGraph = ev.keyGraph

            self.docTitles.extend(ev.docs[key].segTitle for key in ev.docs.keys())
            self.startTimestamp = ev.get_start_timestamp()
            self.endTimestamp = ev.get_end_timestamp()
        else:
            self.startTimestamp = -1
            self.endTimestamp = -1

        self.hotness = 0.0
        self.age = 0
        self.staleAge = 0
        self.summary = ""
        self.graphEdges = {}

    def get_ID(self):
        max_id = StoryTree.MAX_ID
        StoryTree.MAX_ID += 1
        return max_id

    def isEmpty(self):
        return self.root is None

    def get_number_of_nodes(self, node: StoryNode | None = None):
        if node:
            return self.auxiliary_get_number_of_nodes(node) + 1
        return self.auxiliary_get_number_of_nodes(self.root) + 1 if self.root else 0

    def auxiliary_get_number_of_nodes(self, node: StoryNode):
        numberOfNodes = node.number_of_children()
        for child in node.children:
            numberOfNodes += self.auxiliary_get_number_of_nodes(child)
        return numberOfNodes

    def get_number_of_docs(self):
        return len(self.docTitles)

    def get_number_of_docs_by_time(self, timestamp1: int, timestamp2: int):
        startTimestamp = 0
        endTimestamp = 0
        if timestamp1 < timestamp2:
            startTimestamp = timestamp1
            endTimestamp = timestamp2
        else:
            startTimestamp = timestamp2
            endTimestamp = timestamp1

        numberOfDocs = 0
        storyNodes = self.build(self.root, TreeTraversalOrderEnum.PRE_ORDER)
        for i in range(1, len(storyNodes)):
            sn = self.storyNodes[i]
            for d in sn.e.docs.values():
                if d.publish_time.getTime() >= startTimestamp and d.publish_time.getTime() <= endTimestamp:
                    numberOfDocs += 1

        return numberOfDocs

    def get_start_timestamp(self):
        storyNodes = self.build(self.root, TreeTraversalOrderEnum.PRE_ORDER)
        if len(storyNodes) <= 1:
            return -1
        timestamp = time.time()
        for i in range(1, len(storyNodes)):
            sn = storyNodes[i]
            if sn.e.get_start_timestamp() < timestamp:
                timestamp = sn.e.get_start_timestamp()
        return timestamp

    def get_end_timestamp(self):
        storyNodes = self.build(self.root, TreeTraversalOrderEnum.PRE_ORDER)
        if len(storyNodes) <= 1:
            return -1
        timestamp = -1
        for i in range(1, len(storyNodes)):
            sn = storyNodes[i]
            if sn.e.get_end_timestamp() > timestamp:
                timestamp = sn.e.get_end_timestamp()
        return timestamp

    def build(self, traversalOrder):
        returnList = None
        if self.root is not None:
            returnList = self.build(self.root, traversalOrder)
        return returnList

    def build(self, node: StoryNode, traversalOrder):
        traversalResult = []

        if traversalOrder == TreeTraversalOrderEnum.PRE_ORDER:
            self.build_pre_order(node, traversalResult)

        elif traversalOrder == TreeTraversalOrderEnum.POST_ORDER:
            self.build_post_order(node, traversalResult)
        return traversalResult

    def build_pre_order(self, node: StoryNode, traversalResult):
        traversalResult.add(node)

        for child in node.children:
            self.build_pre_order(child, traversalResult)

    # TODO: to remove is not really used
    def build_post_order(self, node: StoryNode, traversalResult):
        for child in node.children:
            self.build_post_order(child, traversalResult)
        traversalResult.add(node)


class StoryForest:
    def __init__(self) -> None:
        self.storyTrees = []
        self.corpus = Corpus()
        self.cumulativeDF = {}
        self.cumulativeDocAmount = 0

    def get_all_events(self):
        result = []
        for st in self.storyTrees:
            storyNodes = st.build(TreeTraversalOrderEnum.PRE_ORDER)
            result.extend(storyNodes[i].e for i in range(1, len(storyNodes)))
        return result

    def filter_story_trees_by_time(self, t):
        result = []
        t = datetime.fromtimestamp(t)
        for st in self.storyTrees:
            storyEndTime = datetime.fromtimestamp(st.endTimestamp)
            if storyEndTime < t:
                result.append(st)
        return result


class StoryMaker:
    def __init__(self, corpus: Corpus, model) -> None:
        self.corpus = corpus
        self.model = model

    # generate stories from corpus
    def generate_stories(self, events) -> StoryForest:
        sf = StoryForest()
        # events = extract_events_from_corpus(self.corpus)
        sf = self.update_stories_by_events(sf, events)
        return sf

    # this could be done daily
    def update_stories_by_events(self, sf: StoryForest, events: list[Event]) -> StoryForest:
        for e in events:
            storyIdx = self.find_related_story(e, sf)
            if storyIdx >= 0:
                self.update_story_tree(sf, storyIdx, e)
                sf.storyTrees[storyIdx].staleAge = -1
            else:
                newSt = StoryTree(e)
                newSt.staleAge = -1
                sf.storyTrees.append(newSt)

        # we increase the age and stale age of each story tree
        for idx in range(0, len(sf.storyTrees)):
            sf.storyTrees[idx].age += 1
            sf.storyTrees[idx].staleAge += 1

        return sf

    def update_story_tree(self, sf: StoryForest, storyIdx: int, e: Event):
        st = sf.storyTrees[storyIdx]
        storyNodes = st.build(TreeTraversalOrderEnum.PRE_ORDER)

        maxCompatibility = -1
        matchIdx = -1
        # sameEvent = False

        # compare with each story node
        for i in range(1, len(storyNodes)):
            sameEv = self.same_event(e, storyNodes[i], len(sf.corpus.docs))
            if sameEv:
                matchIdx = i
                break

            # if not an existing event, calculate compatibility
            compatibility = self.calc_compatibility_event_2story_node(e, storyNodes[i], st)
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
        st.keyGraph = KeywordGraph.merge_key_graphs(st.keyGraph, e.keyGraph)
        if st.startTimestamp > e.get_end_timestamp():
            st.startTimestamp = e.get_start_timestamp
        if st.endTimestamp < e.get_end_timestamp:
            st.endTimestamp = e.get_end_timestamp()

        for d in e.docs.values():
            st.docTitles.append(d.segTitle)

    def find_related_story(self, e: Event, sf: StoryForest) -> int:
        matchIdx = -1
        for i in range(0, len(sf.storyTrees)):
            if self.same_story_by_rule(e, sf, i):
                matchIdx = i
                break
        return matchIdx

    def same_story_by_rule(self, e: Event, sf: StoryForest, storyTreeIdx: int) -> bool:
        st = sf.storyTrees[storyTreeIdx]
        eventDocTitles = [d.title for d in e.docs]
        if eventDocTitles in {d.segTitle for d in e.docs.values() if d.segTitle in st.docTitles}:
            return True

        # use sone rules for brand new event
        # compare event and story's keyword graphs
        keyGraphCompatibility = self.calc_keygraph_compatibility_event_2story(e, st)
        if keyGraphCompatibility < MinKeyGraphCompatibilityEv2St:
            return False

        # Remove stop words
        english_stopwords = stopwords.words("english")
        german_stopwords = stopwords.words("german")
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

    def calc_keygraph_compatibility_event_2story(self, e: Event, st: StoryTree) -> float:
        numIntersection = len(set(e.keyGraph.graphNodes.keys()).intersection(set(st.keyGraph.graphNodes.keys())))
        numUnion = len(e.keyGraph.graphNodes.keys()) + len(st.keyGraph.graphNodes.keys()) - numIntersection
        return (numIntersection + 0.0) / numUnion if numUnion > 0 else 0

    def calc_compatibility_event_2story_node(self, e: Event, sn: StoryNode, st: StoryTree) -> float:
        compatibility = 0
        if e.centroid is None:
            e.calc_centroid()
        if sn.e.centroid is None:
            sn.e.calc_centroid()

        # content similarity
        event2StoryNodeCompatibility = Document.cosine_similarity_by_tf(e.centroid, sn.e.centroid)

        # path similarity
        event2PathCompatibility = (event2StoryNodeCompatibility + (sn.numPathNode - 1) * sn.consistency) / sn.numPathNode

        # time proximity
        timeProximity = 0
        T = abs(max(st.endTimestamp, e.get_end_timestamp()) - min(st.startTimestamp, e.get_start_timestamp()))
        timeGap = 0
        if T != 0:
            timeGap = (e.get_start_timestamp() - sn.e.get_start_timestamp()) / (T + 0.0)

        if timeGap >= 0:
            timeProximity = math.exp(-timeGap * DeltaTimeGap)
        else:
            timeProximity = 0 - math.exp(timeGap * DeltaTimeGap)

        return event2StoryNodeCompatibility * event2PathCompatibility * timeProximity

    # TODO: check if correct and if it matches the description in the paper
    def merge(self, e: Event, sn: StoryNode):
        sn.e.docs.update(e.docs)
        sn.startTimestamp = sn.e.get_start_timestamp()
        sn.endTimestamp = sn.e.get_end_timestamp()

    def extend(self, e: Event, sn: StoryNode):
        if len(e.docs) > 0:
            newSn = StoryNode(e)
            sn.addChild(newSn)
            if not sn.isRoot():
                event2StoryNodeCompatibility = Document.cosine_similarity_by_tf(e.centroid, sn.e.centroid)
                event2PathCompatibility = (event2StoryNodeCompatibility + (sn.numPathNode - 1) * sn.consistency) / sn.numPathNode
                newSn.consistency = event2PathCompatibility
            else:
                newSn.consistency = 0

    def same_event(self, e: Event, sn: StoryNode, model) -> bool:
        # check if duplicated docs
        eDocTitles = set()
        for d in e.docs.values():
            eDocTitles.add(d.segTitle)

        snDocTitles = set()
        for d in sn.e.docs.values():
            snDocTitles.add(d.segTitle)

        snDocTitles = {t for t in snDocTitles if t in eDocTitles}
        if len(snDocTitles) > 0:
            return True

        # get the first document in each document cluster
        keyd1 = next(iter(e.docs))
        keyd2 = next(iter(sn.e.docs))

        d1 = e.docs[keyd1]
        d2 = sn.e.docs[keyd2]

        return same_event(d1, d2, self.model)
