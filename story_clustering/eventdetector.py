import math
import json
import torch
from sentence_transformers import SentenceTransformer, util
from .keywords_organizer import KeywordGraph, CommunityDetector, KeywordNode
from .document_representation import Keyword, Document, Corpus
from .event_organizer import Event
from .nlp_utils import tfidf, idf


EventSplitAlg = "DocGraph"
MinTopicSize = 1



def extract_events_incrementally(corpus:Corpus, g: KeywordGraph, model) -> list[Event]:
    # extract keyword communities from keyword graph
    calc_docs_tfidf_vector_size_with_graph(corpus.docs, corpus.DF, g.graphNodes)

    communities = CommunityDetector(g.graphNodes).detect_communities_louvain()
    #s = len(communities)
    #print(f"Communities size: {s}")

    # extract events from corpus based on keyword communities
    events = extract_topic_by_keyword_communities(corpus, communities)
    print("docs to events assigned. Detecting sub-events...")

    # identify more fine-grained events
    events = split_events(events, model)

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
    print(f"Communities size: {s}")

    # extract events from corpus based on keyword communities
    events = extract_topic_by_keyword_communities(corpus, communities)
    print("docs to events assigned. Detecting sub-events...")

    # identify more fine-grained events
    events = split_events(events, model)

    for event in events:
        event.refine_key_graph()

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
        for doc in corpus.docs.values():
            cosineSimilarity = tfidf_cosine_similarity_graph_2doc(c, doc, corpus.DF, len(corpus.docs))
            if cosineSimilarity > doc_similarity[doc.doc_id]:
                doc_community[doc.doc_id] = i
                doc_similarity[doc.doc_id] = cosineSimilarity

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


def split_events(events: list[Event], model) -> list[Event]:
    # updated the original implementation to use sentence transformers to detect if
    # two documents talk about the same event
    result = []
    for e in events:
        if len(e.docs) >= 2:
            split_events_list = []
            processed_doc_keys = []

            for i, d1 in enumerate(e.docs.keys()):
                if d1 in processed_doc_keys:
                    continue
                processed_doc_keys.append(d1)
                sub_event = Event()
                sub_event.keyGraph = KeywordGraph()
                sub_event.keyGraph.graphNodes = e.keyGraph.graphNodes.copy()
                sub_event.docs[d1] = e.docs[d1]
                sub_event.similarities[d1] = e.similarities[d1]

                for j in range(i + 1, len(e.docs.keys())):
                    d2 = list(e.docs.keys())[j]
                    if d2 not in processed_doc_keys and same_event(e.docs[d1], e.docs[d2],  model):
                        sub_event.docs[d2] = e.docs[d2]
                        sub_event.similarities[d2] = e.similarities[d2]
                        processed_doc_keys.append(d2)

                split_events_list.append(sub_event)
            result.extend(split_events_list)
        else:
            result.append(e)
    return result


def compute_similarity(text_1, text_2, model):
    sent_text_1 = text_1.replace("\n", " ").split(".")
    sent_text_2 = text_2.replace("\n", " ").split(".")

    sent_text_2 = [s for s in sent_text_2 if s != ""][:10]
    sent_text_1 = [s for s in sent_text_1 if s != ""][:10]

    em_1 = model.encode(sent_text_1, convert_to_tensor=True)
    em_2 = model.encode(sent_text_2, convert_to_tensor=True)

    consine_sim_1 = util.pytorch_cos_sim(em_1, em_2)
    max_vals, _inx = torch.max(consine_sim_1, dim=1)
    avg = torch.mean(max_vals, dim=0)
    return avg.item()


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
            # print(type(k))
            keyword = Keyword(baseform=k["baseForm"], words=k["words"], tf=k["tf"], df=k["df"])
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


if __name__ == "__main__":
    # read and create Corpus
    f = open("awake/data/corpus_test.json", "r")
    # print(corpus)
    corpus_dict = json.load(f)
    f.close()
    print("Corpus loaded...")
    corpus = create_corpus_from_json(corpus_dict)
    corpus.update_df()

    print("Corpus created...")
    # extract events from corpus
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    events = extract_events_from_corpus(corpus=corpus, model=model)
    for e in events:
        print("Titles:")
        for d in e.docs:
            print(e.docs[d].title)
        print("------")
        print(f"Keywords: {list(e.keyGraph.graphNodes.keys())}")
        print("-------------")
