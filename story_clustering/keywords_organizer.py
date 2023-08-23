import math
import time
import itertools
import networkx as nx
from networkx.algorithms.community.quality import modularity
from story_clustering.document_representation import Corpus, Keyword

# from multiprocessing import Pool

# MinEdgeDF = 2
MinEdgeDF = 1
MinEdgeCorrelation = 0.05
MaxClusterNodeSize = 5
MinClusterNodeSize = 1
MinCpToDuplicateEdge = 0.2


class KeywordNode:
    """
    A class to define the vertex for keyword graph
    Attributes
    ------------
    id: int
        Node identifier
    keyword: Keyword
        Keyword contained in third node
    edges: dict (id -> KeywordEdge)
        Edges connected with this node
    prev: KeywordNode
        Previously visited node
    visited: bool
        Whether this node has been visited or not

    Methods
    ------------
    insertEdge(KeywordNode n2)
        connects this node to new node n2
    removeEdge(KeywordNode n2)
        removes edge that connects this node to node n2
    removeAllEdges()
        removes all edges for this node
    """

    maxID = 0

    def __init__(self, keyword) -> None:
        self.id = self.maxID
        self.keyword = keyword
        self.edges = {}
        self.prev = None
        self.visited = False

        # autoincrement maxID
        self.maxID += 1


class KeywordEdge:
    """
    A class to represent an edge in a keyword graph
    Attributes
    -------------
    id: str
        Edge identifier
    n1: KeywordNode
    n2: KeywordNode
        Connected nodes
    df: int
        how many times this edge exists within all documents
    cp1: double
        conditional probability p(n2 | n1)
    cp2: double
        conditional probability p(n1 | n2)
    betweennessScore: double
        betweenness score

    Methods
    -------------
    get_id(KeywordNode n1, KeywordNode n2):
        generate edge id from two connected nodes
    increase_df():
        increase document frequency of the edge
    compute_cps():
        update edge's two conditional probabilities
    opposite(KeywordNode n):
        given one end node of the edge, return the node at the other end
    compare_betweenness(KeywordEdge e): int
        compare two edges's betweenness score: -1 if edge e has higher betweenness score (less important)
        1 if edge e has lower betweenness score (more important)
        0 if the two edges are of the same
    compare_edge_strength(KeywordEdge e): int
        compare two edges' link strength: -1 denotes the edge to be compared with is less important,
        1 denotes the edge to be compared with is more important,
        0 denotes their strengths are the same.
    """

    def __init__(self, n1: KeywordNode, n2: KeywordNode, id) -> None:
        self.n1 = n1
        self.n2 = n2
        self.id = id
        self.df = 0
        self.cp2 = None
        self.cp1 = None
        self.betweennessScore = None

    def increase_df(self):
        self.df += 1

    @staticmethod
    def get_id(n1: KeywordNode, n2: KeywordNode) -> str:
        if n1.keyword.baseForm < n2.keyword.baseForm:
            return f"{n1.keyword.baseForm}_{n2.keyword.baseForm}"
        return f"{n2.keyword.baseForm}_{n1.keyword.baseForm}"

    def compute_cps(self):
        self.cp1 = 1.0 * self.df / self.n1.keyword.df
        self.cp2 = 1.0 * self.df / self.n2.keyword.df

    def opposite(self, n: KeywordNode):
        if self.n1.keyword.baseForm == n.keyword.baseForm:
            return self.n2
        return self.n1 if self.n2.keyword.baseForm == n.keyword.baseForm else None

    def compare_betweenness(self, e) -> int:
        if len(self.n1.edges) < 2 or len(self.n2.edges) < 2 or self.betweennessScore < e.betweennessScore:
            return -1
        if self.betweennessScore > e.betweennessScore:
            return 1
        if self.df > e.df:
            return -1
        return 1 if self.df < e.df else 0

    def compare_edge_strength(self, e) -> int:
        cp = max(self.cp1, self.cp2)
        ecp = max(e.cp1, e.cp2)

        if cp > ecp:
            return -1
        if cp < ecp:
            return 1
        if self.df > e.df:
            return -1
        return 1 if self.df < e.df else 0

    def get_edge_strength(self) -> float:
        return max(self.cp1, self.cp2)


class KeywordGraph:
    """
    This class defines a keyword graph (KeyGraph)
    Attributes
    ---------------
    graphNodes: the map (str, KeywordNode) of graph nodes
    Methods
    ---------------
    build_graph(Corpus): builds the graph for an input corpus of documents
    merge_key_graphs(dict): merges graph g into current one
    removeNode(str): removes node
    getKeywords(): return all string keywords in the graph
    graphToJSON(): returns existing graph as JSON string
    """

    def __init__(self):
        self.graphNodes = {}

    def build_graph(self, corpus: Corpus):
        # create nodes for each keyword in the corpus
        # all_keywords = set()
        for document in corpus.docs.values():
            for keyword in document.keywords.values():
                node = None
                if keyword.baseForm in self.graphNodes:
                    node = self.graphNodes[keyword.baseForm]
                else:
                    new_keyword = Keyword(keyword.baseForm, keyword.words, keyword.documents, tf=0, df=corpus.DF.get(keyword.baseForm))
                    # new_keyword.df = corpus.DF[new_keyword.baseForm]
                    node = KeywordNode(keyword=new_keyword)
                    self.graphNodes[keyword.baseForm] = node

                node.keyword.documents.add(document.doc_id)
                # the node term frequency is updated keeping in mind the term frequency in each document
                node.keyword.increase_tf(keyword.tf)
        # create edges between co-occurring keywords
        for document in corpus.docs.values():
            for keyword1 in document.keywords.values():
                if keyword1.baseForm in self.graphNodes:
                    node1 = self.graphNodes[keyword1.baseForm]
                    for keyword2 in document.keywords.values():
                        if keyword2.baseForm in self.graphNodes and keyword1.baseForm < keyword2.baseForm:
                            node2 = self.graphNodes[keyword2.baseForm]
                            edgeId = KeywordEdge.get_id(node1, node2)
                            if edgeId not in node1.edges:
                                new_edge = KeywordEdge(node1, node2, edgeId)
                                new_edge.df += 1
                                node1.edges[edgeId] = new_edge
                                node2.edges[edgeId] = new_edge
                            else:
                                node1.edges[edgeId].df += 1
                                ## HERE TO CHECK IF THE SAME EDGE IS IN node2 or we need to do it manually for it as well
                                if node2.edges[edgeId].df != node1.edges[edgeId].df:
                                    print("!!! Edge object is not the same")
                                    node2.edges[edgeId].df = node1.edges[edgeId].df
                            # e = node1.edges[edgeId]
                            # print(f'Doc_id: {document.url}, k1: {keyword1.baseForm}, k2: {keyword2.baseForm}, edge_df: {e.df}  ')

        # filter edges
        to_remove = []
        for node in self.graphNodes.values():
            for edge in node.edges.values():
                # if (edge.df >= 3):
                # print(f'Values edgeDF:{edge.df}, n1DF: {edge.n1.keyword.df}, n2DF: {edge.n2.keyword.df}')
                # remove edges with small df or edges with samll edge correlation (which means node n1 n2
                # may also be connected with a lot of other nodes)
                MI = edge.df / (edge.n1.keyword.df + edge.n2.keyword.df - edge.df)
                if edge.df < MinEdgeDF or MI < MinEdgeCorrelation:
                    to_remove.append(edge)
                else:
                    edge.compute_cps()
            for edge in to_remove:
                edge.n1.edges.pop(edge.id)
                edge.n2.edges.pop(edge.id)
            to_remove.clear()

        for nodekey in list(self.graphNodes.keys()):
            if len(self.graphNodes[nodekey].edges) == 0:
                self.graphNodes.pop(nodekey)

    @staticmethod
    def merge_key_graphs(kg1: dict[str, KeywordNode], kg2: dict[str, KeywordNode]) -> dict[str, KeywordNode]:
        kg = {n.keyword.baseForm: KeywordNode(n.keyword) for n in kg1.values()}
        for node in kg2.values():
            kg[node.keyword.baseForm] = KeywordNode(node.keyword)

        for node in kg1.values():
            for edge in node.edges.values():
                if (
                    node.keyword.baseForm.compareTo(edge.opposite(node).keyword.baseForm) < 0
                    and edge.id not in kg[edge.n1.keyword.baseForm].edges
                ):
                    n1 = kg[edge.n1.keyword.baseForm]
                    n2 = kg[edge.n2.keyword.baseForm]
                    ee = KeywordEdge(n1, n2, edge.id)
                    n1.edges[ee.id] = ee
                    n2.edges[ee.id] = ee

        for node in kg2.values():
            for edge in node.edges.values():
                if (
                    node.keyword.baseForm.compareTo(edge.opposite(node).keyword.baseForm) < 0
                    and edge.id not in kg[edge.n1.keyword.baseForm].edges
                ):
                    n1 = kg[edge.n1.keyword.baseForm]
                    n2 = kg[edge.n2.keyword.baseForm]
                    ee = KeywordEdge(n1, n2, edge.id)
                    n1.edges[ee.id] = ee
                    n2.edges[ee.id] = ee

        return kg


class CommunityDetector:
    """
    A class to extract graph communities from a keyword graph
    Attributes:
    ------------
    nodes: a map (keyword, KeywordNode)
        Graph to detect keywords communities
    communities: a map (keyword, KeywordNode)
        Keyword communities
    Methods:
    ------------
    detectCommunities(): returns a list of sub-graphs representing keywords communities
    find_connected_components(nodes): returns a list of sub-graphs representing the connected components
    filter_top_k_percent_of_edges(nodes,k): filters top k percentage of edges in a graph
    """

    def __init__(self, nodes: dict) -> None:
        self.nodes = nodes
        # self.communities = self.detectCommunities()

    def detect_communities_louvain(self) -> dict:
        gr = nx.Graph()
        keywords_dict = {}
        for i, n in enumerate(self.nodes.values(), start=1):
            keywords_dict[i] = n.keyword.baseForm
            gr.add_node(i)
        keywords_vals = {v: k for (k, v) in keywords_dict.items()}
        for w1 in gr.nodes():
            word = keywords_dict[w1]
            for e in self.nodes[word].edges.values():
                gr.add_edge(keywords_vals[e.n1.keyword.baseForm], keywords_vals[e.n2.keyword.baseForm], weight=e.df)

        gr_copy = gr.copy()
        start = time.time()
        communities = nx.community.louvain_communities(gr_copy, seed=123)
        # communities = nx.community.girvan_newman(G)
        end = time.time()
        print(f"Time to louvain communities alg (sec): {end-start}")

        key_communities = []
        for c in communities:
            if len(c) > 1:
                subgraph = gr.subgraph(c)
                key_communities.append(subgraph)

        return self.get_keywords_keygraphs(key_communities, keywords_dict)

    def get_keywords_keygraphs(self, communities, keywords_dict):
        all_communities = []

        for subgraph in communities:
            new_keywords_graph = KeywordGraph()
            for i in subgraph.nodes():
                word = keywords_dict[i]
                existing_node = self.nodes[word]
                keyword_node = KeywordNode(existing_node.keyword)
                new_keywords_graph.graphNodes[keyword_node.keyword.baseForm] = keyword_node
            for u, v, weight in subgraph.edges(data=True):
                w1 = keywords_dict[u]
                w2 = keywords_dict[v]
                edge_id = KeywordEdge.get_id(new_keywords_graph.graphNodes[w1], new_keywords_graph.graphNodes[w2])
                keyword_edge = KeywordEdge(new_keywords_graph.graphNodes[w1], new_keywords_graph.graphNodes[w2], id=edge_id)
                keyword_edge.df = weight["weight"]
                new_keywords_graph.graphNodes[w1].edges[edge_id] = keyword_edge
                new_keywords_graph.graphNodes[w2].edges[edge_id] = keyword_edge
            all_communities.append(new_keywords_graph)
        return all_communities

    def filter_top_k_percent_of_edges(self, nodes, k) -> bool:
        edgeSize = sum(len(n.edges) for n in nodes.values())
        edgeSize /= 2

        ntoremove = int((edgeSize * k) / 100)
        if ntoremove == 0:
            return False

        # order edges based on their strength/significance
        all_edges = list({e for n in nodes.values() for e in n.edges.values()})
        all_edges.sort(key=lambda x: max(x.cp1, x.cp2))

        for e in all_edges[:ntoremove]:
            e.n1.edges.pop(e.id)
            e.n2.edges.pop(e.id)

        return True

    @DeprecationWarning
    def insert_into(self, toRemove, e):
        # if list is of length 1
        if len(toRemove) == 1:
            if toRemove[0] is not None and toRemove[0].compare_edge_strength(e) >= 0:
                return False

            toRemove[0] = e
            return True
        if len(toRemove) > 0 and toRemove[len(toRemove) - 1].compare_edge_strength(e) >= 0:
            return False
        i = len(toRemove) - 1 if len(toRemove) > 0 else 0
        while i >= 1 and (toRemove[i - 1] is None or toRemove[i - 1].compare_edge_strength(e) < 0):
            toRemove[i] = toRemove[i - 1]
            i -= 1
        toRemove.insert(i, e)
        return True
