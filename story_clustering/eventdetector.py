import math
import torch
from torch import Tensor
from sentence_transformers import util
from story_clustering.keywords_organizer import KeywordGraph, CommunityDetector, KeywordNode
from story_clustering.document_representation import Document, Corpus
from story_clustering.event_organizer import Event
from story_clustering.nlp_utils import tfidf, idf, get_sentence_transformer
from story_clustering.log import logger
import numpy as np

SIMILARITY_THRESHOLD = 0.5
DIFF_SIMILARITY_THRESHOLD = 0.3


def extract_events_from_corpus(corpus: Corpus, graph: KeywordGraph | None = None) -> list[Event]:
    if not graph:
        graph = KeywordGraph()
        graph.build_graph(corpus)

    # extract keyword communities from keyword graph
    calc_docs_tfidf_vector_size_with_graph(corpus.docs, corpus.df, graph.graph_nodes)

    communities = CommunityDetector(graph.graph_nodes).detect_communities_louvain()
    logger.info(f"Number of communities: {len(communities)}")
    return extract_topic_by_keyword_communities(corpus, communities)


def calc_docs_tfidf_vector_size_with_graph(docs: dict[str, Document], df: dict[str, int], graph_nodes: dict[str, KeywordNode]):
    # calculate the tfidf value for each document in docs
    # the individual tfidf values are the norm of the docs tfidf-vector i.e. (tfidf(keyword1), tfidif(keyword2), ...)
    # consider only keywords that exist as nodes in the graph
    for d in docs.values():
        d.tfidf_vector_size_with_keygraph = sum(
            math.pow(tfidf(k.tf, idf(df[k.baseform], len(docs))), 2) for k in d.keywords.values() if k.baseform in graph_nodes
        )
        d.tfidf_vector_size_with_keygraph = math.sqrt(d.tfidf_vector_size_with_keygraph)


def calc_docs_tfidf_vector_size_with_graph_2(docs: dict[str, Document], df: dict[str, int], communities: list[KeywordGraph]):
    # calculate the tfidf value for each document in docs
    # the individual tfidf values are the norm of the docs keywords tfidf-vector i.e. (tfidf(keyword1), tfidif(keyword2), ...)
    # each keyword is counted once for each community it appears in

    for d in docs.values():
        d.tfidf_vector_size_with_keygraph = sum(
            math.pow(tfidf(k.tf, idf(df[k.baseform], len(docs))), 2)
            for k in d.keywords.values()
            for graph in communities
            if k.baseform in graph.graph_nodes
        )
        d.tfidf_vector_size_with_keygraph = math.sqrt(d.tfidf_vector_size_with_keygraph)


def extract_topic_by_keyword_communities(corpus: Corpus, communities: list[KeywordGraph], doc_size=None) -> list[Event]:
    result = []
    if doc_size is None:
        doc_size = len(corpus.docs)

    # for each document, get the similarities with every community
    doc_comm_similarities: dict[str, list] = {
        doc.doc_id: [tfidf_cosine_similarity_graph_2doc(comm, doc, corpus.df, doc_size) for comm in communities]
        for doc in corpus.docs.values()
    }

    # create an event out of each community, add the docs that are most similar to it
    for idx, community in enumerate(communities):
        logger.info(f"Processing community {idx}/{len(communities)}")
        event = Event(key_graph=community)

        for doc in corpus.docs.values():
            if np.argmax(doc_comm_similarities[doc.doc_id]).item() == idx:
                event.docs[doc.doc_id] = doc
                event.similarities[doc.doc_id] = doc_comm_similarities[doc.doc_id][idx]
                doc.processed = True

        logger.info(f"Community {idx}/{len(communities)} - contains {len(event.docs)}")
        result.extend(split_events_incr_clustering(event))

    return result


def tfidf_cosine_similarity_graph_2doc(community: KeywordGraph, d2: Document, df: dict[str, int], doc_size: int) -> float:
    sim = 0
    vectorsize1 = 0

    for node in community.graph_nodes.values():
        # calculate community keyword's tf
        n_tf = 0
        for edge in node.edges.values():
            edge.compute_cps()
            n_tf += max(edge.cp1, edge.cp2)
        node.keyword.tf = n_tf / len(node.edges) if len(node.edges) > 0 else 0

        if node.keyword.baseform in df:
            # update vector size of community
            vectorsize1 += math.pow(tfidf(node.keyword.tf, idf(df[node.keyword.baseform], doc_size)), 2)

            # update similarity between document d2 and community
            if node.keyword.baseform in d2.keywords:
                sim += tfidf(node.keyword.tf, idf(df[node.keyword.baseform], doc_size)) * tfidf(
                    d2.keywords[node.keyword.baseform].tf, idf(df[node.keyword.baseform], doc_size)
                )
    vectorsize1 = math.sqrt(vectorsize1)

    # return similarity
    if vectorsize1 > 0 and d2.tfidf_vector_size_with_keygraph > 0:
        return sim / vectorsize1 / d2.tfidf_vector_size_with_keygraph
    return 0


def split_events_incr_clustering(event: Event) -> list[Event]:
    split_events_list = []
    processed_doc_keys = set()
    e_docs_ids_list = list(event.docs.keys())

    if len(event.docs) == 1:
        if event.key_graph.story_id is not None:
            doc_id = list(event.docs.keys())[0]
            if not same_event_cluster(event.docs[doc_id], event.key_graph.text):
                event.key_graph.story_id = None
                event.key_graph.text = ""

        event.refine_key_graph()
        return [event]

    for i, d1 in enumerate(e_docs_ids_list):
        if d1 in processed_doc_keys:
            continue

        processed_doc_keys.add(d1)
        sub_event = Event(key_graph=event.key_graph or None)
        sub_event.docs[d1] = event.docs[d1]
        sub_event.similarities[d1] = event.similarities[d1]

        if event.key_graph.story_id is not None and not same_event_cluster(sub_event.docs[d1], event.key_graph.text):
            sub_event.key_graph.story_id = None
            sub_event.key_graph.text = ""

        for d2 in e_docs_ids_list[i + 1 :]:
            if d2 in processed_doc_keys:
                continue
            if sub_event.key_graph.story_id is None:
                if same_event(sub_event.docs[d1], event.docs[d2]):
                    sub_event.docs[d2] = event.docs[d2]
                    sub_event.similarities[d2] = event.similarities[d2]
                    processed_doc_keys.add(d2)
            else:
                if same_new_event(sub_event.docs[d1], event.docs[d2], sub_event.key_graph.text):
                    sub_event.key_graph.story_id = None
                    sub_event.key_graph.text = ""
                    sub_event.docs[d2] = event.docs[d2]
                    sub_event.similarities[d2] = event.similarities[d2]
                    processed_doc_keys.add(d2)

        sub_event.refine_key_graph()
        split_events_list.append(sub_event)

    return split_events_list


def split_events(event: Event) -> list[Event]:
    if len(event.docs) < 2:
        event.refine_key_graph()
        return [event]

    split_events_list = []
    processed_doc_keys = set()
    e_docs_ids_list = list(event.docs.keys())

    for i, d1 in enumerate(e_docs_ids_list):
        if d1 in processed_doc_keys:
            continue

        processed_doc_keys.add(d1)
        sub_event = Event(key_graph=event.key_graph or None)
        sub_event.docs[d1] = event.docs[d1]
        sub_event.similarities[d1] = event.similarities[d1]

        for d2 in e_docs_ids_list[i + 1 :]:
            if d2 in processed_doc_keys:
                continue
            if sub_event.key_graph.story_id is None:
                if same_event(sub_event.docs[d1], event.docs[d2]):
                    sub_event.docs[d2] = event.docs[d2]
                    sub_event.similarities[d2] = event.similarities[d2]
                    processed_doc_keys.add(d2)
            else:
                if same_new_event(sub_event.docs[d1], event.docs[d2], sub_event.key_graph.text):
                    sub_event.key_graph.story_id = None
                    sub_event.key_graph.text = ""
                sub_event.docs[d2] = event.docs[d2]
                sub_event.similarities[d2] = event.similarities[d2]
                processed_doc_keys.add(d2)

        sub_event.refine_key_graph()
        split_events_list.append(sub_event)

    return split_events_list


def compute_similarity(text_1: str, text_2: str) -> float:
    sent_text_1 = text_1.replace("\n", " ").split(".")
    sent_text_2 = text_2.replace("\n", " ").split(".")

    sent_text_2 = [f"{s}." for s in sent_text_2 if s != ""][:5]
    sent_text_1 = [f"{s}." for s in sent_text_1 if s != ""][:5]

    em_1 = Tensor(get_sentence_transformer().encode(sent_text_1, convert_to_tensor=True, show_progress_bar=False))
    em_2 = Tensor(get_sentence_transformer().encode(sent_text_2, convert_to_tensor=True, show_progress_bar=False))

    consine_sim_1 = util.pytorch_cos_sim(em_1, em_2)
    max_vals, _inx = torch.max(consine_sim_1, dim=1)
    avg = torch.mean(max_vals, dim=0)
    return avg.item()


def same_event_text(text_1, text_2) -> bool:
    return compute_similarity(text_1, text_2) >= SIMILARITY_THRESHOLD


def same_event_cluster(d1: Document, cluster_text: str) -> bool:
    text_1 = d1.content
    if not cluster_text or not text_1:
        return False
    return compute_similarity(text_1, cluster_text) >= SIMILARITY_THRESHOLD


def same_event(d1: Document, d2: Document) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    if not text_1 or not text_2:
        return False
    return compute_similarity(text_1, text_2) >= SIMILARITY_THRESHOLD


def same_new_event(d1: Document, d2: Document, keygraph_text: str) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    if not text_1 or not text_2:
        return False
    sim_1 = compute_similarity(text_1, text_2)
    sim_2 = compute_similarity(text_2, keygraph_text)
    return (sim_1 - sim_2) >= DIFF_SIMILARITY_THRESHOLD
