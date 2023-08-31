from sentence_transformers import SentenceTransformer, util
import torch
from .document_representation import Keyword, Document, Corpus
from .event_organizer import Event
from .eventdetector import extract_events_from_corpus, extract_events_incrementally
from .keywords_organizer import KeywordGraph, KeywordEdge, KeywordNode
from .nlp_utils import compute_tf
import time

EventSplitAlg = "DocGraph"
MinTopicSize = 1
SimilarityThreshold = 0.44


def reformat_string(s: str) -> str:
    return (s.lower().replace('ä', 'ae').replace('ö', 'oe')
            .replace('ü', 'ue').replace('ß', "ss"))


def create_corpus(new_news_items: list[dict]) -> Corpus:
    """Creates a Corpus object from a JSON object denoting all documents

    Args:
        new_news_items (list[dict]): list of dict with following keys
            {'id': str, 'link': str, 'text': str, 'title':str,'date': 'YYYY-MM-DD', 'lang': str,
            'tags': [{'baseForm': str, 'words': list[str]}]}
    Returns:
        corpus: Corpus of documents
    """
    corpus = Corpus()
    for nitem_agg in new_news_items:
        for nitem in nitem_agg["news_items"]:
            doc = Document(doc_id=nitem["id"])

            doc.url = nitem.get("news_item_data.link", None)
            doc.content = nitem["news_item_data"]["content"] or nitem["news_item_data"]["review"]
            if not doc.content:
                continue
            doc.title = nitem["news_item_data"]["title"]
            if doc.title is not None:
                doc.segTitle = doc.title.strip().split(" ")
            doc.publish_time = nitem.get("news_item_data.published", None)
            doc.language = nitem["news_item_data"]["language"]
            # print(doc.doc_id)
            # create keywords
            keywords = {}
            for tag in nitem_agg["tags"].values():
                keyword = Keyword(
                    baseform=reformat_string(tag["name"]), words=tag["sub_forms"], tf=tag.get("tf", 0), df=tag.get("df", 0),
                    documents=None
                )
                keywords[reformat_string(tag["name"])] = keyword

                # print(tag["name"])
                # update tf for keyword
                if keyword.baseForm not in doc.content:
                    continue

                if keyword.tf == 0:
                    keyword.tf = compute_tf(keyword.baseForm, keyword.words, doc.content)
            doc.keywords = keywords
            corpus.docs[doc.doc_id] = doc

    corpus.update_df()
    return corpus


def initial_clustering(new_news_items: list):
    # create corpus
    corpus = create_corpus(new_news_items)

    # extract events
    # this is too slow model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    events = extract_events_from_corpus(corpus=corpus, model=model)

    # create stories based on events
    print(f"Aggregating events into stories...")
    t1 = time.time()
    stories = []
    for event in events:
        found_story = False
        for story in stories:
            if belongs_to_story(event, story, model):
                story.append(event)
                found_story = True
                break
        if not found_story:
            aux = [event]
            stories.append(aux)
    t2 = time.time()
    print(f"Time to construct stories (sec): {t2-t1}")

    new_aggregates = to_json_events(events)
    new_aggregates = new_aggregates | to_json_stories(stories)
    # new_aggregates.update(to_json_stories(stories))
    return new_aggregates


def to_json_events(events: list[Event]) -> dict:
    # iterate over each event and return the list of documents
    # ids belonging to the same event
    all_events = []
    for event in events:
        all_events.append(list(event.docs.keys()))
    return {"event_clusters": all_events}


def to_json_stories(stories: list[list[Event]]) -> dict:
    # iterate over each story
    # iterate over each event in story
    all_stories = []
    for story in stories:
        s_docs = []
        for event in story:
            s_docs.extend(list(event.docs))
        all_stories.append(s_docs)
    return {"story_clusters": all_stories}


def get_or_add_keywordNode(tag: dict, graphNodes: dict) -> KeywordNode:
    if reformat_string(tag["name"]) in graphNodes:
        return graphNodes[reformat_string(tag["name"])]

    keyword = Keyword(baseform=reformat_string(tag["name"]), words=tag["sub_forms"], documents=None, tf=tag.get("tf", 0), df=tag.get("df", 0))
    keywordNode = KeywordNode(keyword=keyword)
    graphNodes[keyword.baseForm] = keywordNode
    return keywordNode


def update_or_create_keywordEdge(kn1: KeywordNode, kn2: KeywordNode):
    edgeId = KeywordEdge.get_id(kn1, kn2)
    if edgeId not in kn1.edges:
        new_edge = KeywordEdge(kn1, kn2, edgeId)
        new_edge.df += 1
        kn1.edges[edgeId] = new_edge
        kn2.edges[edgeId] = new_edge
    else:
        kn1.edges[edgeId].df += 1

        if kn1.edges[edgeId].df != kn2.edges[edgeId].df:
            print("Edge object is not the same")
            kn2.edges[edgeId].df = kn1.edges[edgeId].df


def incremental_clustering(new_news_items: list, already_clusterd_events: list):
    # create corpus from news_items
    corpus = create_corpus(new_news_items)

    # create keyGraph from corpus
    g = KeywordGraph()
    g.build_graph(corpus=corpus)

    # add to g the new nodes and edges from already_clusterd_events
    for cluster in already_clusterd_events:
        tags = cluster["tags"]
        for keyword_1 in tags.values():
            for keyword_2 in tags.values():
                if keyword_1 != keyword_2:
                    keyNode1 = get_or_add_keywordNode(keyword_1, g.graphNodes)
                    keyNode2 = get_or_add_keywordNode(keyword_2, g.graphNodes)
                    # add edge and increase edge df
                    update_or_create_keywordEdge(keyNode1, keyNode2)

    # extract events
    # model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    events = extract_events_incrementally(corpus=corpus, g=g, model=model)

    # create stories based on events
    stories = []
    for event in events:
        found_story = False
        for story in stories:
            if belongs_to_story(event, story, model):
                story.append(event)
                found_story = True
                break
        if not found_story:
            aux = [event]
            stories.append(aux)

    new_aggregates = to_json_events(events)
    new_aggregates = new_aggregates | to_json_stories(stories)
    return new_aggregates


def compute_similarity_for_stories(text_1, text_2, model):
    sent_text_1 = text_1.replace("\n", " ").split(".")
    sent_text_2 = text_2.replace("\n", " ").split(".")

    sent_text_2 = [s for s in sent_text_2 if s != ""]
    sent_text_1 = [s for s in sent_text_1 if s != ""]

    em_1 = model.encode(sent_text_1, convert_to_tensor=True, show_progress_bar=False)
    em_2 = model.encode(sent_text_2, convert_to_tensor=True, show_progress_bar=False)

    consine_sim_1 = util.pytorch_cos_sim(em_1, em_2)
    max_vals, _inx = torch.max(consine_sim_1, dim=1)
    avg = torch.mean(max_vals, dim=0)
    return avg.item()


def belongs_to_story(ev, story, model) -> bool:
    text_1 = " ".join([d.title for d in ev.docs.values()])
    text_2 = " ".join([d.title for e in story for d in e.docs.values()])
    # print(text_1)
    # print(text_2)
    return compute_similarity_for_stories(text_1, text_2, model) >= SimilarityThreshold
