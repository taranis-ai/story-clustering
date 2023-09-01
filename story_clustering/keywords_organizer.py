import networkx as nx
from story_clustering.document_representation import Corpus, Keyword

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

    def __init__(self, n1: KeywordNode, n2: KeywordNode, id: str) -> None:
        self.n1 = n1
        self.n2 = n2
        self.id: str = id
        self.df: int = 0
        self.cp2: float = 0
        self.cp1: float = 0
        self.betweennessScore: float = 0

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
    """

    def __init__(self):
        self.graphNodes = {}

    def build_graph(self, corpus: "Corpus"):
        self.graphNodes = {}

        def get_or_create_node(keyword: "Keyword") -> "KeywordNode":
            if keyword.baseForm not in self.graphNodes:
                new_keyword = Keyword(keyword.baseForm, keyword.words, keyword.documents, tf=0, df=corpus.DF.get(keyword.baseForm, 0))
                self.graphNodes[keyword.baseForm] = KeywordNode(keyword=new_keyword)
            return self.graphNodes[keyword.baseForm]

        def filter_and_remove_edges():
            to_remove: list["KeywordEdge"] = []
            for node in self.graphNodes.values():
                for edge in list(node.edges.values()):
                    MI = edge.df / (edge.n1.keyword.df + edge.n2.keyword.df - edge.df)
                    if edge.df < MinEdgeDF or MI < MinEdgeCorrelation:
                        to_remove.append(edge)
            for edge in to_remove:
                edge.n1.edges.pop(edge.id, None)
                edge.n2.edges.pop(edge.id, None)

        # Create or update nodes and edges
        for document in corpus.docs.values():
            for k1 in document.keywords.values():
                node1 = get_or_create_node(k1)
                node1.keyword.documents.add(document.doc_id)
                node1.keyword.increase_tf(k1.tf)

                for k2 in document.keywords.values():
                    if k1.baseForm < k2.baseForm:
                        node2 = get_or_create_node(k2)
                        edge_id = KeywordEdge.get_id(node1, node2)
                        edge = node1.edges.get(edge_id, KeywordEdge(node1, node2, edge_id))
                        edge.df += 1
                        node1.edges[edge_id] = edge
                        node2.edges[edge_id] = edge

        filter_and_remove_edges()
        self.graphNodes = {k: v for k, v in self.graphNodes.items() if v.edges}


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
    """

    def __init__(self, nodes: dict) -> None:
        self.nodes = nodes
        # self.communities = self.detectCommunities()

    def detect_communities_louvain(self) -> list[KeywordGraph]:
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
        communities = nx.community.louvain_communities(gr_copy, seed=123)
        # communities = nx.community.girvan_newman(G)

        key_communities = []
        for c in communities:
            if len(c) > 1:
                subgraph = gr.subgraph(c)
                key_communities.append(self.get_keywords_keygraphs(subgraph, keywords_dict))

        return key_communities

    def get_keywords_keygraphs(self, subgraph, keywords_dict):
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
        return new_keywords_graph
