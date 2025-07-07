from .document_representation import Document, Keyword
from .keywords_organizer import KeywordGraph
from datetime import datetime


class Event:
    def __init__(self, key_graph: KeywordGraph):
        self.key_graph = key_graph
        self.max_id: int = 1
        self.docs: dict[str, Document] = {}
        self.similarities = {}
        self.centroid: Document = Document("-1")

    def calc_centroid(self):
        # calculate the centroid document of this document cluster
        # centroid is the concatenation of all docs in this event

        timestamp = datetime.max
        for doc in self.docs.values():
            if doc.publish_time < timestamp:
                timestamp = doc.publish_time
            for k in doc.keywords.values():
                if self.centroid.keywords and k.baseform in self.centroid.keywords:
                    kk = self.centroid.keywords[k.baseform]
                    kk.tf += k.tf
                    kk.df += k.df
                else:
                    self.centroid.set_keyword(Keyword(k.baseform, set(), k.tf, k.df))

        self.centroid.calc_tf_vector_size()
        self.centroid.publish_time = timestamp

    def refine_key_graph(self):
        # remove all of the events key_graphs keywords that do not appear in any of the events docs

        to_remove = []
        if not self.key_graph:
            return

        to_remove.extend(
            keyword for keyword in self.key_graph.graph_nodes.keys() if not any(d.contains_keyword(keyword) for d in self.docs.values())
        )
        for keyword in to_remove:
            self.key_graph.graph_nodes.pop(keyword)
