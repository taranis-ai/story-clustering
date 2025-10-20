import networkx as nx
from story_clustering.document_representation import Corpus, Keyword

MIN_EDGE_DF = 1
MIN_EDGE_CORRELATION = 0.05
MAX_CLUSTER_NODE_SIZE = 5
MIN_CLUSTER_NODE_SIZE = 1
MIN_CP_TO_DUPLICATE_EDGE = 0.2


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

    max_id: int = 0

    def __init__(self, keyword: Keyword):
        self.keyword = keyword
        self.id: int = self.max_id
        self.edges: dict[str, KeywordEdge] = {}
        self.prev: KeywordNode | None = None
        self.visited: bool = False

        # autoincrement max_id
        self.max_id += 1


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

    def __init__(self, n1: KeywordNode, n2: KeywordNode, id: str):
        self.n1 = n1
        self.n2 = n2
        self.id = id
        self.df: int = 0
        self.cp2: float = 0
        self.cp1: float = 0
        self.betweenness_score: float = 0

    def increase_df(self):
        self.df += 1

    @staticmethod
    def get_id(n1: KeywordNode, n2: KeywordNode) -> str:
        if n1.keyword.baseform < n2.keyword.baseform:
            return f"{n1.keyword.baseform}_{n2.keyword.baseform}"
        return f"{n2.keyword.baseform}_{n1.keyword.baseform}"

    def compute_cps(self):
        self.cp1 = 1.0 * self.df / self.n1.keyword.df
        self.cp2 = 1.0 * self.df / self.n2.keyword.df

    def opposite(self, n: KeywordNode) -> KeywordNode | None:
        if self.n1.keyword.baseform == n.keyword.baseform:
            return self.n2
        return self.n1 if self.n2.keyword.baseform == n.keyword.baseform else None

    def compare_betweenness(self, e: "KeywordEdge") -> int:
        if len(self.n1.edges) < 2 or len(self.n2.edges) < 2 or self.betweenness_score < e.betweenness_score:
            return -1
        if self.betweenness_score > e.betweenness_score:
            return 1
        if self.df > e.df:
            return -1
        return 1 if self.df < e.df else 0

    def compare_edge_strength(self, e: "KeywordEdge") -> int:
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

    def __init__(self, story_id: str | None = None):
        self.story_id = story_id
        self.graph_nodes: dict[str, KeywordNode] = {}
        self.text: str = ""

    def build_graph(self, corpus: "Corpus"):
        self.graph_nodes = {}

        def get_or_create_node(keyword: Keyword) -> KeywordNode:
            if keyword.baseform not in self.graph_nodes:
                new_keyword = Keyword(baseform=keyword.baseform, documents=set(), tf=0, df=corpus.df.get(keyword.baseform, 0))
                self.graph_nodes[keyword.baseform] = KeywordNode(keyword=new_keyword)
            return self.graph_nodes[keyword.baseform]

        def filter_and_remove_edges():
            to_remove: list[KeywordEdge] = []
            for node in self.graph_nodes.values():
                for edge in node.edges.values():
                    # mutual_information = edge.df / (edge.n1.keyword.df + edge.n2.keyword.df)
                    mutual_information = edge.df / (corpus.df[edge.n1.keyword.baseform] + corpus.df[edge.n2.keyword.baseform])
                    if edge.df < MIN_EDGE_DF or mutual_information < MIN_EDGE_CORRELATION:
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
                    if k1.baseform < k2.baseform:
                        node2 = get_or_create_node(k2)
                        node2.keyword.documents.add(document.doc_id)
                        node2.keyword.increase_tf(k2.tf)
                        edge_id = KeywordEdge.get_id(node1, node2)
                        edge = node1.edges.get(edge_id, KeywordEdge(node1, node2, edge_id))
                        edge.df += 1
                        node1.edges[edge_id] = edge
                        node2.edges[edge_id] = edge

        filter_and_remove_edges()
        self.graph_nodes = {k: v for k, v in self.graph_nodes.items() if len(v.edges) > 0}


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

    def __init__(self, nodes: dict[str, KeywordNode]):
        self.nodes = nodes
        # self.communities = self.detectCommunities()

    def detect_communities_louvain(self) -> list[KeywordGraph]:
        gr = nx.Graph()
        keywords_dict = {}
        for i, n in enumerate(self.nodes.values(), start=1):
            keywords_dict[i] = n.keyword.baseform
            gr.add_node(i)
        keywords_vals = {v: k for (k, v) in keywords_dict.items()}
        for w1 in gr.nodes():
            word = keywords_dict[w1]
            for e in self.nodes[word].edges.values():
                gr.add_edge(keywords_vals[e.n1.keyword.baseform], keywords_vals[e.n2.keyword.baseform], weight=e.df)

        # check if there are any

        gr_copy = gr.copy()
        communities = nx.community.louvain_communities(gr_copy, seed=42, resolution=1)  # type: ignore
        # communities = nx.community.girvan_newman(G)

        key_communities = []
        for c in communities:
            if len(c) > 1:
                subgraph = nx.Graph(gr.subgraph(c))
                subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
                # print(f"Print nodes: {subgraph.nodes()}")
                # print(f"Print edges: {subgraph.edges()}")
                key_communities.append(self.get_keywords_keygraphs(subgraph, keywords_dict))
        return key_communities

    def get_keywords_keygraphs(self, subgraph: nx.Graph, keywords_dict: dict[int, str]) -> KeywordGraph:
        new_keywords_graph = KeywordGraph()
        for i in subgraph.nodes():
            word = keywords_dict[i]
            existing_node = self.nodes[word]
            keyword_node = KeywordNode(existing_node.keyword)
            new_keywords_graph.graph_nodes[keyword_node.keyword.baseform] = keyword_node
        for u, v, weight in subgraph.edges(data=True):
            w1 = keywords_dict[u]
            w2 = keywords_dict[v]
            edge_id = KeywordEdge.get_id(new_keywords_graph.graph_nodes[w1], new_keywords_graph.graph_nodes[w2])
            keyword_edge = KeywordEdge(new_keywords_graph.graph_nodes[w1], new_keywords_graph.graph_nodes[w2], id=edge_id)
            keyword_edge.df = weight["weight"]
            new_keywords_graph.graph_nodes[w1].edges[edge_id] = keyword_edge
            new_keywords_graph.graph_nodes[w2].edges[edge_id] = keyword_edge
        return new_keywords_graph
