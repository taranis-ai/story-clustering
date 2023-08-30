import math
import torch
from sentence_transformers import util
from .keywords_organizer import KeywordGraph, CommunityDetector, KeywordNode
from .document_representation import Keyword, Document, Corpus
from .event_organizer import Event
from .nlp_utils import tfidf, idf
import time
import networkx as nx


EventSplitAlg = "DocGraph"
MinTopicSize = 1


def extract_events_incrementally(corpus: Corpus, g: KeywordGraph, model) -> list[Event]:
    # extract keyword communities from keyword graph
    calc_docs_tfidf_vector_size_with_graph(corpus.docs, corpus.DF, g.graphNodes)

    communities = CommunityDetector(g.graphNodes).detect_communities_louvain()
    # s = len(communities)
    # print(f"Communities size: {s}")

    # extract events from corpus based on keyword communities
    events = extract_topic_by_keyword_communities(corpus, communities)
    print("Docs to events assigned. Detecting sub-events...")

    # identify more fine-grained events
    s1 = time.time()
    events = split_events(events, model)
    s2 = time.time()
    print(f"Split events (sec): {s2-s1}")

    print("Sub-events detected. Refining event keygraph...")
    for event in events:
        event.refine_key_graph()

    return events


def extract_events_from_corpus(corpus: Corpus, model) -> list[Event]:
    graph = KeywordGraph()
    graph.build_graph(corpus)

    # extract keyword communities from keyword graph
    calc_docs_tfidf_vector_size_with_graph(corpus.docs, corpus.DF, graph.graphNodes)

    communities = CommunityDetector(graph.graphNodes).detect_communities_louvain()
    s = len(communities)
    print(f"Communities size: {s}. Assigning docs to communities...")

    # extract events from corpus based on keyword communities
    t1 = time.time()
    events = extract_topic_by_keyword_communities(corpus, communities)
    t2 = time.time()
    print(f"Time to assign docs to communities (sec): {t2-t1}. ")
    print("Docs to events assigned. Detecting sub-events...")

    # identify more fine-grained events
    # t1 = time.time()
    # events = split_events_opmitized(events, model)
    # t2 = time.time()
    # print(f"Split events optimized (sec): {t2-t1} ")

    s1 = time.time()
    events = split_events(events, model)
    s2 = time.time()
    print(f"Split events (sec): {s2-s1}")

    print("Sub-events detected. Refining event keygraph...")

    for event in events:
        event.refine_key_graph()

    print("Keygraph refined. Returning events.")

    return events


def calc_docs_tfidf_vector_size_with_graph(docs: dict[str, Document], DF: dict[str, float], graphNodes: dict[str, KeywordNode]):
    for d in docs.values():
        d.tfidfVectorSizeWithKeygraph = sum(
            math.pow(tfidf(k.tf, idf(DF[k.baseForm], len(docs))), 2) for k in d.keywords.values() if k.baseForm in graphNodes
        )
        d.tfidfVectorSizeWithKeygraph = math.sqrt(d.tfidfVectorSizeWithKeygraph)


def extract_topic_by_keyword_communities(corpus: Corpus, communities: dict[str, KeywordNode]) -> list[Event]:
    result = []
    doc_community = {}
    doc_similarity = {}
    # initialization
    for d in corpus.docs.values():
        doc_community[d.doc_id] = -1
        doc_similarity[d.doc_id] = -1.0

    for i, c in enumerate(communities):
        # t1 = time.time()
        for doc in corpus.docs.values():
            # start = time.time()
            cosineSimilarity = tfidf_cosine_similarity_graph_2doc(c, doc, corpus.DF, len(corpus.docs))
            # end = time.time()
            # print(f"    Time to compute doc-comunity similarity (sec): {end-start}")
            if cosineSimilarity > doc_similarity[doc.doc_id]:
                doc_community[doc.doc_id] = i
                doc_similarity[doc.doc_id] = cosineSimilarity
        # t2 = time.time()
        # print(f"For comunity {i} time to compute all similarities (sec): {t2-t1}")

    # create event for each community
    for i, c in enumerate(communities):
        e = Event()
        e.keyGraph = c

        for doc_id in doc_community:
            if doc_community[doc_id] == i:
                d = corpus.docs[doc_id]
                if d.doc_id not in e.docs:
                    e.docs[d.doc_id] = d
                    e.similarities[d.doc_id] = doc_similarity[doc_id]
                d.processed = True

        if len(e.docs) >= MinTopicSize:
            result.append(e)
    return result


def tfidf_cosine_similarity_graph_2doc(community: dict[str, KeywordNode], d2: Document, DF: dict[str, float], docSize: int) -> float:
    sim = 0
    vectorsize1 = 0
    number_of_keywords_in_common = 0

    for n in community.graphNodes.values():
        # calculate community keyword's tf
        nTF = 0
        for e in n.edges.values():
            e.compute_cps()
            nTF += max(e.cp1, e.cp2)
        n.keyword.tf = nTF / len(n.edges)

        if n.keyword.baseForm in DF:
            # update vector size of community
            vectorsize1 += math.pow(tfidf(n.keyword.tf, idf(DF[n.keyword.baseForm], docSize)), 2)

            # update similarity between document d2 and community
            if n.keyword.baseForm in d2.keywords:
                number_of_keywords_in_common += 1
                sim += tfidf(n.keyword.tf, idf(DF[n.keyword.baseForm], docSize)) * tfidf(
                    d2.keywords[n.keyword.baseForm].tf, idf(DF[n.keyword.baseForm], docSize)
                )
    vectorsize1 = math.sqrt(vectorsize1)

    # return similarity
    if vectorsize1 > 0 and d2.tfidfVectorSizeWithKeygraph > 0:
        return sim / vectorsize1 / d2.tfidfVectorSizeWithKeygraph
    return 0


def split_events_by_community_detection(events: list[Event], model) -> list[Event]:
    result = []
    for e in events:
        if len(e.docs) >= 2:
            e_docs_ids_list = list(e.docs.keys())
            n_docs = len(e_docs_ids_list)
            print(f"Number of documents in the event: {n_docs}")
            edges = []
            start = time.time()

            edges.extend(
                [
                    (i, j)
                    for i in range(n_docs)
                    for j in range(n_docs)
                    if i < j and same_event(e.docs[e_docs_ids_list[i]], e.docs[e_docs_ids_list[j]], model)
                ]
            )
            end = time.time()
            G = nx.Graph()
            G.add_nodes_from([*range(0, n_docs, 1)])
            G.add_edges_from(edges)
            S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            print(f"Compute connected components (sec): {end - start}")
            for conn_comp in S:
                sub_event = Event()
                sub_event.keyGraph = KeywordGraph()
                sub_event.keyGraph.graphNodes = e.keyGraph.graphNodes.copy() if e.keyGraph else None
                for id in conn_comp:
                    sub_event.docs[e_docs_ids_list[id]] = e.docs[e_docs_ids_list[id]]
                    sub_event.similarities[e_docs_ids_list[id]] = e.similarities[e_docs_ids_list[id]]
                result.append(sub_event)

        else:
            result.append(e)
    return result


def split_events(events: list[Event], model) -> list[Event]:
    # updated the original implementation to use sentence transformers to detect if
    # two documents talk about the same event
    result = []
    for e in events:
        if len(e.docs) >= 2:
            split_events_list = []
            processed_doc_keys = []
            e_docs_ids_list = list(e.docs.keys())
            for i, d1 in enumerate(e_docs_ids_list):
                if d1 in processed_doc_keys:
                    continue
                processed_doc_keys.append(d1)
                sub_event = Event()
                sub_event.keyGraph = KeywordGraph()
                sub_event.keyGraph.graphNodes = e.keyGraph.graphNodes.copy() if e.keyGraph else None
                sub_event.docs[d1] = e.docs[d1]
                sub_event.similarities[d1] = e.similarities[d1]

                for j in range(i + 1, len(e.docs.keys())):
                    d2 = e_docs_ids_list[j]
                    if d2 not in processed_doc_keys and same_event(e.docs[d1], e.docs[d2], model):
                        sub_event.docs[d2] = e.docs[d2]
                        sub_event.similarities[d2] = e.similarities[d2]
                        processed_doc_keys.append(d2)

                split_events_list.append(sub_event)
            result.extend(split_events_list)
        else:
            result.append(e)
    return result


def compute_similarity(text_1, text_2, model):
    # start = time.time()
    sent_text_1 = text_1.replace("\n", " ").split(".")
    sent_text_2 = text_2.replace("\n", " ").split(".")

    sent_text_2 = [s for s in sent_text_2 if s != ""][:5]
    sent_text_1 = [s for s in sent_text_1 if s != ""][:5]

    em_1 = model.encode(sent_text_1, convert_to_tensor=True, show_progress_bar=False)
    em_2 = model.encode(sent_text_2, convert_to_tensor=True, show_progress_bar=False)

    consine_sim_1 = util.pytorch_cos_sim(em_1, em_2)
    max_vals, _inx = torch.max(consine_sim_1, dim=1)
    avg = torch.mean(max_vals, dim=0)
    # end = time.time()
    # print(f"    Time to compute doc-doc similarity (sec): {end - start}")
    return avg.item()


def same_event_text(text_1, text_2, model):
    return compute_similarity(text_1, text_2, model) >= 0.44


def same_event(d1: Document, d2: Document, model) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    return compute_similarity(text_1, text_2, model) >= 0.44


def create_corpus_from_json(corpus_dict) -> Corpus:
    docs = {}
    DF = {}
    corpus = Corpus()

    for doc_id, doc in corpus_dict["docs"].items():
        keywords = {}

        for k in doc["keywords"]:
            keyword = Keyword(baseform=k["baseForm"], words=k["words"], documents=None, tf=k["tf"], df=k["df"])
            keywords[k["baseForm"]] = keyword
        corpus.docs[doc_id] = Document(
            doc_id=doc_id,
            url=doc["url"],
            publish_date=doc["publish_time"],
            language=doc["language"],
            title=doc["title"],
            content=doc["content"],
            keywords=keywords,
        )

    return corpus
