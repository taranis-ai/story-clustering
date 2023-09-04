import math
import torch
from torch import Tensor
from sentence_transformers import util
from collections import defaultdict
from .keywords_organizer import KeywordGraph, CommunityDetector, KeywordNode
from .document_representation import Document, Corpus
from .event_organizer import Event
from .nlp_utils import tfidf, idf
from story_clustering import logger, sentence_transformer
import numpy as np


def extract_events_from_corpus(corpus: Corpus, graph: KeywordGraph | None = None) -> list[Event]:
    if not graph:
        graph = KeywordGraph()
        graph.build_graph(corpus)

    # extract keyword communities from keyword graph
    calc_docs_tfidf_vector_size_with_graph(corpus.docs, corpus.DF, graph.graphNodes)

    communities = CommunityDetector(graph.graphNodes).detect_communities_louvain()
    logger.info(f"Number of communities: {len(communities)}")
    return extract_topic_by_keyword_communities(corpus, communities)


def calc_docs_tfidf_vector_size_with_graph(docs: dict[str, Document], DF: dict[str, float], graphNodes: dict[str, KeywordNode]):
    for d in docs.values():
        d.tfidfVectorSizeWithKeygraph = sum(
            math.pow(tfidf(k.tf, idf(DF[k.baseForm], len(docs))), 2) for k in d.keywords.values() if k.baseForm in graphNodes
        )
        d.tfidfVectorSizeWithKeygraph = math.sqrt(d.tfidfVectorSizeWithKeygraph)


def extract_topic_by_keyword_communities(corpus: Corpus, communities: list) -> list[Event]:
    result = []

    max_comm = {
        doc.doc_id: np.argmax([tfidf_cosine_similarity_graph_2doc(community, doc, corpus.DF, len(corpus.docs)) for community in communities])
        for doc in corpus.docs.values()
    }
    for i, community in enumerate(communities):
        logger.info(f"Processing community {i}/{len(communities)}")
        event = process_community(i, community, corpus, max_comm)
        logger.info(f"Community {i}/{len(communities)} - contains {len(event.docs)}")
        result.extend(split_events(event))
    return result


def process_community(i: int, community: KeywordGraph, corpus: Corpus, max_comm: dict):
    event = Event()
    event.keyGraph = community
    doc_similarity: dict[str, float] = defaultdict(lambda: -1.0)

    for doc in corpus.docs.values():
        cosineSimilarity = tfidf_cosine_similarity_graph_2doc(community, doc, corpus.DF, len(corpus.docs))
        if max_comm[doc.doc_id] > doc_similarity[doc.doc_id]:
            doc_similarity[doc.doc_id] = cosineSimilarity
            d = corpus.docs[doc.doc_id]
            if d.doc_id not in event.docs:
                event.docs[d.doc_id] = d
                event.similarities[d.doc_id] = doc_similarity[doc.doc_id]
            d.processed = True
    return event


def tfidf_cosine_similarity_graph_2doc(community: KeywordGraph, d2: Document, DF: dict[str, float], docSize: int) -> float:
    sim = 0
    vectorsize1 = 0
    number_of_keywords_in_common = 0

    for node in community.graphNodes.values():
        # calculate community keyword's tf
        nTF = 0
        for edge in node.edges.values():
            edge.compute_cps()
            nTF += max(edge.cp1, edge.cp2)
        node.keyword.tf = nTF / len(node.edges)

        if node.keyword.baseForm in DF:
            # update vector size of community
            vectorsize1 += math.pow(tfidf(node.keyword.tf, idf(DF[node.keyword.baseForm], docSize)), 2)

            # update similarity between document d2 and community
            if node.keyword.baseForm in d2.keywords:
                number_of_keywords_in_common += 1
                sim += tfidf(node.keyword.tf, idf(DF[node.keyword.baseForm], docSize)) * tfidf(
                    d2.keywords[node.keyword.baseForm].tf, idf(DF[node.keyword.baseForm], docSize)
                )
    vectorsize1 = math.sqrt(vectorsize1)

    # return similarity
    if vectorsize1 > 0 and d2.tfidfVectorSizeWithKeygraph > 0:
        return sim / vectorsize1 / d2.tfidfVectorSizeWithKeygraph
    return 0


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
        sub_event = Event(keyGraph=event.keyGraph or None)
        sub_event.docs[d1] = event.docs[d1]
        sub_event.similarities[d1] = event.similarities[d1]

        for d2 in e_docs_ids_list[i + 1 :]:
            if d2 in processed_doc_keys:
                continue
            if same_event(sub_event.docs[d1], event.docs[d2]):
                sub_event.docs[d2] = event.docs[d2]
                sub_event.similarities[d2] = event.similarities[d2]
                processed_doc_keys.add(d2)

        sub_event.refine_key_graph()
        split_events_list.append(sub_event)

    return split_events_list


def compute_similarity(text_1, text_2):
    sent_text_1 = text_1.replace("\n", " ").split(".")
    sent_text_2 = text_2.replace("\n", " ").split(".")

    sent_text_2 = [s for s in sent_text_2 if s != ""][:5]
    sent_text_1 = [s for s in sent_text_1 if s != ""][:5]

    em_1 = Tensor(sentence_transformer.encode(sent_text_1, convert_to_tensor=True, show_progress_bar=False))
    em_2 = Tensor(sentence_transformer.encode(sent_text_2, convert_to_tensor=True, show_progress_bar=False))

    consine_sim_1 = util.pytorch_cos_sim(em_1, em_2)
    max_vals, _inx = torch.max(consine_sim_1, dim=1)
    avg = torch.mean(max_vals, dim=0)
    return avg.item()


def same_event_text(text_1, text_2):
    return compute_similarity(text_1, text_2) >= 0.44


def same_event(d1: Document, d2: Document) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    return compute_similarity(text_1, text_2) >= 0.44
