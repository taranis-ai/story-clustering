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

SimilarityThreshold = 0.5
DiffSimilarityThreshold = 0.3


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
    for d in docs.values():
        d.tfidfVectorSizeWithKeygraph = sum(
            math.pow(tfidf(k.tf, idf(df[k.baseform], len(docs))), 2) for k in d.keywords.values() if k.baseform in graph_nodes
        )
        d.tfidfVectorSizeWithKeygraph = math.sqrt(d.tfidfVectorSizeWithKeygraph)


def calc_docs_tfidf_vector_size_with_graph_2(docs: dict[str, Document], df: dict[str, int], communities: list[KeywordGraph]):
    for d in docs.values():
        d.tfidfVectorSizeWithKeygraph = sum(
            math.pow(tfidf(k.tf, idf(df[k.baseform], len(docs))), 2)
            for k in d.keywords.values()
            for graph in communities
            if k.baseform in graph.graph_nodes
        )
        d.tfidfVectorSizeWithKeygraph = math.sqrt(d.tfidfVectorSizeWithKeygraph)


def extract_topic_by_keyword_communities(corpus: Corpus, communities: list, doc_size=None) -> list[Event]:
    result = []
    if doc_size is None:
        doc_size = len(corpus.docs)

    max_comm = {
        doc.doc_id: np.argmax([tfidf_cosine_similarity_graph_2doc(community, doc, corpus.df, doc_size) for community in communities])
        for doc in corpus.docs.values()
    }
    for i, community in enumerate(communities):
        logger.info(f"Processing community {i}/{len(communities)}")
        event = process_community(i, community, corpus, max_comm)
        logger.info(f"Community {i}/{len(communities)} - contains {len(event.docs)}")
        result.extend(split_events_incr_clustering(event))
    return result


def process_community(community_id: int, community: KeywordGraph, corpus: Corpus, max_comm: dict):
    event = Event(key_graph=community)
    # doc_similarity: dict[str, float] = defaultdict(lambda: -1.0)

    for doc in corpus.docs.values():
        if max_comm[doc.doc_id] == community_id:
            cosineSimilarity = tfidf_cosine_similarity_graph_2doc(community, doc, corpus.df, len(corpus.docs))
            # doc_similarity[doc.doc_id] = cosineSimilarity
            # d = corpus.docs[doc.doc_id]
            # if d.doc_id not in event.docs:
            event.docs[doc.doc_id] = doc
            event.similarities[doc.doc_id] = cosineSimilarity
            doc.processed = True
    return event


def tfidf_cosine_similarity_graph_2doc(community: KeywordGraph, d2: Document, df: dict[str, int], docSize: int) -> float:
    sim = 0
    vectorsize1 = 0
    number_of_keywords_in_common = 0

    for node in community.graph_nodes.values():
        # calculate community keyword's tf
        nTF = 0
        for edge in node.edges.values():
            edge.compute_cps()
            nTF += max(edge.cp1, edge.cp2)
        if len(node.edges) > 0:
            node.keyword.tf = nTF / len(node.edges)
        else:
            node.keyword.tf = 0
        if node.keyword.baseform in df:
            # update vector size of community
            vectorsize1 += math.pow(tfidf(node.keyword.tf, idf(df[node.keyword.baseform], docSize)), 2)

            # update similarity between document d2 and community
            if node.keyword.baseform in d2.keywords:
                number_of_keywords_in_common += 1
                sim += tfidf(node.keyword.tf, idf(df[node.keyword.baseform], docSize)) * tfidf(
                    d2.keywords[node.keyword.baseform].tf, idf(df[node.keyword.baseform], docSize)
                )
    vectorsize1 = math.sqrt(vectorsize1)

    # return similarity
    if vectorsize1 > 0 and d2.tfidfVectorSizeWithKeygraph > 0:
        return sim / vectorsize1 / d2.tfidfVectorSizeWithKeygraph
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


def compute_similarity(text_1: str, text_2: str):
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


def same_event_text(text_1, text_2):
    return compute_similarity(text_1, text_2) >= SimilarityThreshold


def same_event_cluster(d1: Document, cluster_text: str) -> bool:
    text_1 = d1.content
    if not cluster_text or not text_1:
        return False
    return compute_similarity(text_1, cluster_text) >= SimilarityThreshold


def same_event(d1: Document, d2: Document) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    if not text_1 or not text_2:
        return False
    return compute_similarity(text_1, text_2) >= SimilarityThreshold


def same_new_event(d1: Document, d2: Document, keygraph_text) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    if not text_1 or not text_2:
        return False
    sim_1 = compute_similarity(text_1, text_2)
    sim_2 = compute_similarity(text_2, keygraph_text)
    return (sim_1 - sim_2) >= DiffSimilarityThreshold
