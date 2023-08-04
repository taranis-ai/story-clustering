
from .document_representation import Corpus, Document, Keyword
import math
import networkx as nx 
import time
#from multiprocessing import Pool
import time
from networkx.algorithms.community.quality import modularity
import itertools

MinEdgeDF = 2
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
        Keyword contained in thid node
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
    
    def __init__(self,keyword) -> None:
        self.id = self.maxID
        self.keyword = keyword
        self.edges = dict()
        self.prev = None
        self.visited = False
        
        # autoincrement maxID
        self.maxID +=1 
        
    
        

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
    getId(KeywordNode n1, KeywordNode n2):
        generate edge id from two connected nodes
    increaseDF():
        increase document frequency of the edge
    computeCPs():
        update edge's two conditional probabilities
    opposite(KeywordNode n):
        given one end node of the edge, return the node at the other end
    compareBetweenness(KeywordEdge e): int
        compare two edges's betweenness score: -1 if edge e has higher betweenness score (less important)
        1 if edge e has lower betweenness score (more important)
        0 if the two edges are of the same
    compareEdgeStrength(KeywordEdge e): int
        compare two edges' link strength: -1 denotes the edge to be compared with is less important,
        1 denotes the edge to be compared with is more important,
        0 denotes their strengths are the same.
    """
    def __init__(self, n1: KeywordNode, n2:KeywordNode, id) -> None:
        self.n1 = n1
        self.n2 = n2
        # autogenerate id
        #self.id = self.getId(n1,n2)
        self.id = id
        # initiate df
        self.df = 0
    
    def increaseDF(self):
        self.df += 1
        
    @staticmethod
    def getId(n1: KeywordNode, n2: KeywordNode) -> str:
        if n1.keyword.baseForm < n2.keyword.baseForm:
            return n1.keyword.baseForm+'_'+n2.keyword.baseForm
        else:
            return n2.keyword.baseForm+'_'+n1.keyword.baseForm
    
    def computeCPs(self):
        #self.cp1 = 1.0*self.df / self.n1.keyword.tf
        #self.cp2 = 1.0*self.df / self.n2.keyword.tf
        self.cp1 = 1.0*self.df / self.n1.keyword.df
        self.cp2 = 1.0*self.df / self.n2.keyword.df
        
    def opposite(self,n : KeywordNode):
        if self.n1.keyword.baseForm == n.keyword.baseForm:
            return self.n2
        if self.n2.keyword.baseForm == n.keyword.baseForm:
            return self.n1
        return None
    
    def compareBetweenness(self, e) -> int:
        if len(self.n1.edges) < 2 or len(self.n2.edges) < 2 or self.betweennessScore < e.betweennessScore:
            return -1
        if self.betweennessScore > e.betweennessScore:
            return 1
        if self.df > e.df:
            return -1
        if self.df < e.df:
            return 1
        return 0
    
    def compareEdgeStrength(self, e) -> int:
        cp = max(self.cp1,self.cp2)
        ecp = max(e.cp1,e.cp2)
        
        if cp > ecp:
            return -1
        if cp < ecp:
            return 1
        if self.df > e.df:
            return -1
        if self.df < e.df:
            return 1 
        return 0
    
    def getEdgeStrength(self) -> float:
        return max(self.cp1,self.cp2)


class KeywordGraph:
    """
    This class defines a keyword graph (KeyGraph)
    Attributes
    ---------------
    graphNodes: the map (str, KeywordNode) of graph nodes
    Methods
    ---------------
    buildGraph(Corpus): builds the graph for an input corpus of documents
    mergeKeyGraphs(dict): merges graph g into current one
    removeNode(str): removes node 
    getKeywords(): return all string keywords in the graph
    graphToJSON(): returns existing graph as JSON string
    """
    def __init__(self) -> None:
        self.graphNodes = dict()
    
    def buildGraph(self, corpus: Corpus):
        # create nodes for each keyword in the corpus
        #all_keywords = set()
        for document in corpus.docs.values():
            for keyword in document.keywords.values():
                node = None
                if keyword.baseForm in self.graphNodes:
                    node = self.graphNodes[keyword.baseForm]
                else:
                    new_keyword = Keyword(keyword.baseForm, keyword.words, keyword.documents,tf=0,df=corpus.DF.get(keyword.baseForm))
                    #new_keyword.df = corpus.DF[new_keyword.baseForm]
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
                            edgeId = KeywordEdge.getId(node1,node2)
                            if edgeId not in node1.edges:
                                new_edge = KeywordEdge(node1,node2,edgeId)
                                new_edge.df += 1
                                node1.edges[edgeId] = new_edge
                                node2.edges[edgeId] = new_edge
                            else:
                                node1.edges[edgeId].df += 1 
                            ## HERE TO CHECK IF THE SAME EDGE IS IN node2 or we need to do it manually for it as well
                                if node2.edges[edgeId].df != node1.edges[edgeId].df:
                                    print('!!! Edge object is not the same')
                                    node2.edges[edgeId].df =  node1.edges[edgeId].df
                            #e = node1.edges[edgeId]
                            #print(f'Doc_id: {document.url}, k1: {keyword1.baseForm}, k2: {keyword2.baseForm}, edge_df: {e.df}  ')
        
        # filter edges
        to_remove = list()
        for node in self.graphNodes.values():
            for edge in node.edges.values():
                #if (edge.df >= 3):
                    #print(f'Values edgeDF:{edge.df}, n1DF: {edge.n1.keyword.df}, n2DF: {edge.n2.keyword.df}')
                # remove edges with small df or edges with samll edge correlation (which means node n1 n2 may also be connected with a lot of other nodes)
                MI = edge.df / (edge.n1.keyword.df + edge.n2.keyword.df - edge.df)
                if edge.df < MinEdgeDF or MI < MinEdgeCorrelation:
                    to_remove.append(edge)
                else:
                    edge.computeCPs()
            for e in to_remove:
                e.n1.edges.pop(e.id)
                e.n2.edges.pop(e.id)
            to_remove.clear()
            
        for nodekey in list(self.graphNodes.keys()):
            if len(self.graphNodes[nodekey].edges) == 0:
                self.graphNodes.pop(nodekey)
    
    @staticmethod
    def mergeKeyGraphs( kg1: dict[str,KeywordNode], kg2:dict[str,KeywordNode] ) -> dict[str,KeywordNode]:
        kg = dict()
        for n in kg1.values():
            kg[n.keyword.baseForm] = KeywordNode(n.keyword)
        
        for n in kg2.values():
            kg[n.keyword.baseForm] = KeywordNode(n.keyword)
        
        for n in kg1.values():
            for e in n.edges.values():
                if n.keyword.baseForm.compareTo(e.opposite(n).keyword.baseForm) < 0:
                    if not (e.id in kg[e.n1.keyword.baseForm].edges):
                        n1 = kg[e.n1.keyword.baseForm]
                        n2 = kg[e.n2.keyword.baseForm]
                        ee = KeywordEdge(n1,n2,e.id)
                        n1.edges[ee.id] = ee 
                        n2.edges[ee.id] = ee
        
        for n in kg2.values():
            for e in n.edges.values():
                if n.keyword.baseForm.compareTo(e.opposite(n).keyword.baseForm) < 0:
                    if not (e.id in kg[e.n1.keyword.baseForm].edges):
                        n1 = kg[e.n1.keyword.baseForm]
                        n2 = kg[e.n2.keyword.baseForm]
                        ee = KeywordEdge(n1,n2,e.id)
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
    findConnectedComponents(nodes): returns a list of sub-graphs representing the connected components
    filterTopKPercentOfEdges(nodes,k): filters top k percentage of edges in a graph
    """ 
    def __init__(self, nodes:dict) -> None:
        self.nodes = nodes
        #self.communities = self.detectCommunities()
        
    
    def detectCommunitiesLouvain(self) -> dict:
        gr = nx.Graph()
        keywords_dict = dict()
        i = 1
        for n in self.nodes.values():
            keywords_dict[i] = n.keyword.baseForm
            gr.add_node(i)
            i += 1
        
        keywords_vals = { v:k for (k,v) in keywords_dict.items()}
        for w1 in gr.nodes():
            word = keywords_dict[w1]
            for e in self.nodes[word].edges.values():
                gr.add_edge(keywords_vals[e.n1.keyword.baseForm],keywords_vals[e.n2.keyword.baseForm],weight=e.df)
        
        
        G = gr.copy()
        start = time.time()
        communities = nx.community.louvain_communities(G, seed=123)
        #communities = nx.community.girvan_newman(G)
        end = time.time()
        print(f'Time to run girvan newman (sec): {end-start}')
        
        S = []
        for c in communities:
            if len(c) > 1:
                subgraph = gr.subgraph(c)
                S.append(subgraph) 
    
        return self.get_keywords_keygraphs(S, keywords_dict)
    
    def detectCommunitiesKeyGraph(self) -> dict:
        # create network from keywords graph
        gr = nx.Graph()
        keywords_dict = dict()
        i = 1
        for n in self.nodes.values():
            keywords_dict[i] = n.keyword.baseForm
            gr.add_node(i)
            i += 1
        
        keywords_vals = { v:k for (k,v) in keywords_dict.items()}
        for w1 in gr.nodes():
            word = keywords_dict[w1]
            for e in self.nodes[word].edges.values():
                gr.add_edge(keywords_vals[e.n1.keyword.baseForm],keywords_vals[e.n2.keyword.baseForm],weight=e.df)
        
        
        G = gr.copy()
        start = time.time()
        communities = nx.community.girvan_newman(G, most_valuable_edge=betweenness_centrality_parallel)
        #communities = nx.community.girvan_newman(G)
        end = time.time()
        print(f'Time to run girvan newman (sec): {end-start}')
        start = time.time()
        communities_by_quality = []
        i = 0
        k = 100
        limited = itertools.takewhile(lambda c: len(c) <= k, communities)
        for c in limited:
            #print(c)
            print(f'Community {i}')
            i += 1
            #s = time.time()
            m = modularity(G, c)
            #e = time.time()
            #print(f'Time to compute modularity: {e-s}')
            communities_by_quality.append((c,m))
        #communities_by_quality = [(c, modularity(G, c)) for c in communities]
        end = time.time()
        print(f'Time to run communities detection (sec): {end-start}')
        c_best = sorted([(c, m) for c, m in communities_by_quality], key=lambda x: x[1], reverse=True)
        c_best = c_best[0][0]
        # print(Util.pp(communities_by_quality))
        print("Clusters:", modularity(G, c_best), c_best)
        
        
        S = []
        for c in c_best:
            if len(c) > 1:
                subgraph = gr.subgraph(c)
                S.append(subgraph)        
        
        
        #res = tuple(sorted(c) for c in next(comp))
        #print(res)
        
        
       
        #S = [gr.subgraph(c).copy() for c in components]
        #print(len(S))
        
        # create communities keywords
         
        return self.get_keywords_keygraphs(S, keywords_dict)
    
    
    
    def communitySplits(self, graph,betweenness,keywords_dict):
        
        #while(self.keepRemovingEdges(graph,betweenness)):
            #betweenness = nx.edge_betweenness_centrality(graph)
        max_betweenness = 0
        if (len(betweenness.values()) != 0 ):
            max_betweenness = max(betweenness.values())
        else:
            return graph
        for u,v in betweenness.items():
            if float(v) == max_betweenness:
                print(f'Removing edge: {keywords_dict[u[0]]}_{keywords_dict[u[1]]} with betweenness_score: {max_betweenness}')
                graph.remove_edge(u[0], u[1])
        return graph
    
    
    def keepRemovingEdges(self,graph,betweenness):
        #betweenness = nx.edge_betweenness_centrality(graph, normalized=False)
        if (len(betweenness.values()) != 0 ):
            betweennessScore =  max(betweenness.values())
        else:
            return False
        graphSize = len(graph.nodes())
        possiblePath = min(graphSize * (graphSize - 1)/2, MaxClusterNodeSize * (MaxClusterNodeSize -1)/2)
        threshold = 4.35 * math.log(possiblePath) / math.log(2)+1
        #threshold = threshold / (float(2)/((graphSize-1)*(graphSize-2)))
        #threshold = 2.4 * math.log(possiblePath) / math.log(2)+1
        #threshold = math.log(possiblePath) / math.log(2)+1
        keepDetectCommunity = (graphSize > MinClusterNodeSize and betweennessScore > threshold)
        return keepDetectCommunity
        
        
    
    
    def get_keywords_keygraphs(self,communities, keywords_dict):
        all_commpunities= []

        for subgraph in communities:
            new_keywords_graph = KeywordGraph()
            for i in subgraph.nodes():
                word = keywords_dict[i]
                existing_node = self.nodes[word]
                keyword_node = KeywordNode(existing_node.keyword)
                new_keywords_graph.graphNodes[keyword_node.keyword.baseForm] = keyword_node
            for u,v,weight in subgraph.edges(data=True):
                w1 = keywords_dict[u]
                w2 = keywords_dict[v]
                edge_id = KeywordEdge.getId(new_keywords_graph.graphNodes[w1],new_keywords_graph.graphNodes[w2])
                keyword_edge = KeywordEdge(new_keywords_graph.graphNodes[w1],new_keywords_graph.graphNodes[w2],id=edge_id)
                keyword_edge.df = weight['weight']
                new_keywords_graph.graphNodes[w1].edges[edge_id] = keyword_edge
                new_keywords_graph.graphNodes[w2].edges[edge_id] = keyword_edge
            all_commpunities.append(new_keywords_graph)
        return all_commpunities
                
                    
                    
        
        
            
        
         
    
    def detectCommunities(self) -> dict:
        for n in self.nodes.values():
            n.visited = False
        communities = list()
        #nodes_n = self.nodes
        connectedComponents = self.findConnectedComponents(self.nodes)
        
        # processing each connected component sub-graph
        while len(connectedComponents) != 0:
            subNodes = connectedComponents.pop(0)
            # filter small connected components
            if len(subNodes) >= MinClusterNodeSize:
                # iteratively split big connected components into smaller ones
                # this step is in case the graph is too big, and calculate betweenness score is too time consuming
                if len(subNodes) > MaxClusterNodeSize:
                    self.filterTopKPercentOfEdges(subNodes,1)
                    for n in subNodes.values():
                        n.visited = False
                    connectedComponents[:0] = self.findConnectedComponents(subNodes)
                    #connectedComponents.extend(0, self.findConnectedComponents(subNodes))
                else:
                    self.detectCommunitiesBetweenness(subNodes, communities)
        
        return communities
    
    # 
    def findConnectedComponents(self,nodes:dict) -> list[dict]:
        cc = list()
        iter_nodes = iter(set(nodes.values()))
        while len(nodes) > 0:
            source = next(iter_nodes)
            
            subNodes = dict()
            q = list()
            q.insert(0,source)
            while len(q)>0:
                n = q.pop(0)
                n.visited = True
                nodes.pop(n.keyword.baseForm)
                subNodes[n.keyword.baseForm] = n
                for e in n.edges.values():
                    n2 = e.opposite(n)
                    if not n2.visited:
                        n2.visited = True
                        q.append(n2)
            q.clear()
            cc.append(subNodes)
            #l = len(cc)
            #print(f'{source.keyword.baseForm} we found {l} connected components')
            #if len(subNodes) >= 2:
            #    print(subNodes)
        return cc
    
    def filterTopKPercentOfEdges(self,nodes, k) -> bool:
        # count total number of edges in the graph
        edgeSize = 0
        for n in nodes.values():
            edgeSize += len(n.edges)
        edgeSize /= 2
        
        ntoremove = int((edgeSize * k)/100)
        if ntoremove == 0:
            return False
        
        # order edges based on their strength/significance
        all_edges = list(set(e for n in nodes.values() for e in n.edges.values()))
        all_edges.sort(key= lambda x: max(x.cp1,x.cp2))
        
        #toRemove = list()
        #for n1 in nodes.values():
        #    for e in n1.edges.values():
        #        if n1 == e.n1:
        #            self.insertInto(toRemove,e)
        
        for e in all_edges[:ntoremove]:
            e.n1.edges.pop(e.id)
            e.n2.edges.pop(e.id)
            
    @DeprecationWarning
    def insertInto(self,toRemove,e):
        # if list is of length 1
        if len(toRemove) == 1:
            if toRemove[0] is None or toRemove[0].compareEdgeStrength(e) < 0:
                toRemove[0] = e
                return True
            else:
                return False
        
        if len(toRemove)>0 and toRemove[len(toRemove)-1].compareEdgeStrength(e) >=0:
            return False
        else: 
            if len(toRemove) > 0:
                i = len(toRemove)-1
            else:
                i = 0
            while (i >=1 and (toRemove[i-1] is None or toRemove[i-1].compareEdgeStrength(e) < 0)):
                toRemove[i] = toRemove[i-1]
                i -= 1
            toRemove.insert(i,e)
            return True
    
    # find communities using betweenness centrality
    # returns communities extracted from the graph
    def detectCommunitiesBetweenness(self, nodes: dict(), communities):
        # find the edge with maximum betweenness score
        maxKeywordEdge = self.findMaxEdge(nodes)
        
        # decide whether continue to find sub communities
        if self.getFilterStatus(len(nodes), maxKeywordEdge):
            # remove the edge with maximum betweenness score
            maxKeywordEdge.n1.edges.pop(maxKeywordEdge.id)
            maxKeywordEdge.n2.edges.pop(maxKeywordEdge.id)
        
            # check if the graph is stil connected
            # if yes, iteratively run to find communities
            subgraph1 = self.findSubgraph(maxKeywordEdge.n1,nodes)
            if (len(subgraph1) == len(nodes)):
                return self.detectCommunitiesBetweenness(nodes,communities)
            else:
                # remove a subgraph from the whole graph
                for key in subgraph1:
                    nodes.pop(key)
            
            # duplicate edge if the conditional probability is higher than threshold
            if maxKeywordEdge.cp1 > MinCpToDuplicateEdge:
                k = maxKeywordEdge.n2.keyword
                newn = KeywordNode(Keyword(k.baseForm, k.word,k.tf,k.df))
                e = KeywordEdge(maxKeywordEdge.n1,newn)
                maxKeywordEdge.n1.edges[e.id] = e
                newn.edges[e.id] = e
                subgraph1[k.baseForm] = newn
            
            if maxKeywordEdge.cp2 > MinCpToDuplicateEdge:
                k = maxKeywordEdge.n1.keyword
                newn = KeywordNode(Keyword(k.baseForm,k.word,k.tf,k.df))
                e = KeywordEdge(newn,maxKeywordEdge.n2)
                maxKeywordEdge.n2.edges[e.id] = e
                newn.edges[e.id] = e
                nodes[k.baseForm] = newn
            
            self.detectCommunitiesBetweenness(subgraph1, communities)
            self.detectCommunitiesBetweenness(nodes, communities)
            return communities
        else:
            communities.append(nodes)
            return communities

    def findMaxEdge(self,nodes: dict()):
        # clear each edge's betweenness score as it changes when graph structure changes
        for n in nodes.values():
            for e in n.edges.values():
                e.betweennessScore = 0
        
        maxKewordEdge = KeywordEdge(None,None,None)
        maxKewordEdge.betweennessScore = -1
        
        for source in  nodes.values():
            for n in nodes.values():
                n.visited = False
            maxKewordEdge = self.BFS (source,maxKewordEdge)
        
        # for undirected graph, each shortest path will be count twice as each nide will be retrieved twice as source or destination
        maxKewordEdge.betweennessScore /= 2
        return maxKewordEdge
    
    # using BFS to get edge with maximum betweenness score
    def BFS(self, source: KeywordNode, maxKeywordEdge: KeywordEdge):
        q = list()
        q.append(source)
        
        while len(q) > 0:
            n = q.pop(0)
            for e in n.edges.values():
                n2 = e.opposite(n)
                if not n2.visited:
                    n2.visited = True
                    n2.prev = n
                    self.updateBetweennessScore(n2, source, maxKeywordEdge)
                    if e.compareBetweenness(maxKeywordEdge) > 0:
                        maxKeywordEdge = e
                    q.append(n2)
        return maxKeywordEdge
    
    def updateBetweennessScore(self, n, root, maxKeywordEdge):
        while True:
            e = n.edges.get(KeywordEdge.getId(n,n.prev))
            e.betweennessScore += 1
            if e.compareBetweenness(maxKeywordEdge) > 0:
                maxKeywordEdge = e
            n = n.prev
            if n.id == root.id:
                break
        return maxKeywordEdge
    
    # extract sub-graph that contains a specific node
    def findSubgraph(self, source, nodes):
        for n in nodes.values():
            n.visited = False
        
        subnodes = dict()
        q = list()
        q.append(source)
        while len(q)>0:
            n = q.pop(0)
            n.visited = True
            subnodes[n.keyword.baseForm] = n
            for e in n.edges.values():
                n2 = e.opposite(n)
                if not n2.visited:
                    n2.visited = True
                    q.append(n2)
        
        return subnodes
    
    # decide whether continue to split graph into subnodes
    def getFilterStatus(self, graphSize:int, maxKeywordEdge: KeywordEdge):
        possiblePath = min(graphSize * (graphSize - 1)/2, MaxClusterNodeSize * (MaxClusterNodeSize -1)/2)
        #threshold = 4.2 * math.log(possiblePath) / math.log(2)+1
        threshold = 2.4 * math.log(possiblePath) / math.log(2)+1
        keepDetectCommunity = (graphSize > MinClusterNodeSize and maxKeywordEdge is not None and maxKeywordEdge.df > 0 and maxKeywordEdge.betweennessScore > threshold)
        return keepDetectCommunity
        
    
    
            
        
                
                    
                           
                
        
        
        
        
                                   