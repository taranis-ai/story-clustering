from sentence_transformers import SentenceTransformer, util
import torch
from .document_representation import Keyword, Document, Corpus
from .event_organizer import Event
from .eventdetector import extract_events_from_corpus, extract_events_incrementally
from .keywords_organizer import KeywordGraph, KeywordEdge, KeywordNode
from .nlp_utils import compute_tf
EventSplitAlg = "DocGraph"
MinTopicSize = 1
SimilarityThreshold = 0.44


def create_corpus(new_news_items: list[dict]) -> Corpus:
    """Creates a Corpus object from a JSON object denoting all documents

    Args:
        new_news_items (list[dict]): list of dict with following keys
            {'id': str, 'url': str, 'text': str, 'title':str,'date': 'YYYY-MM-DD', 'lang': str,
            'tags': [{'baseForm': str, 'words': list[str]}]}
    Returns:
        corpus: Corpus of documents
    """
    corpus = Corpus()
    for nitem in new_news_items:
        doc = Document(doc_id=nitem["id"])

        doc.url = nitem.get("url", None)
        doc.content = nitem["text"]
        doc.title = nitem["title"]
        doc.publish_time = nitem.get("date", None)
        doc.language = nitem["lang"]
        # create keywords
        keywords = {}
        for tag in nitem["tags"]:
            keyword = Keyword(baseform=tag["baseform"], words=tag["words"], tf=tag.get("tf", 0), df=tag.get("df", 0))
            keywords[tag["baseForm"]] = keyword

            # update tf for keyword
            if keyword.tf == 0:
                keyword.tf = compute_tf(keyword.baseForm,keyword.words,doc.content)
        doc.keywords = keywords
        corpus.docs[doc.doc_id] = doc

    corpus.update_df()
    return corpus


def initial_clustering(new_news_items: list):
    # create corpus
    corpus = create_corpus(new_news_items)

    # extract events
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    events = extract_events_from_corpus(corpus=corpus, model=model)

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


def to_json_events(events: list[Event]) -> dict:
    # iterate over each event and return the list of documents ids belonging to the same event
    all_events = []
    for event in events:
        all_events.append(event)
    return {"event_clusters": all_events}


def to_json_stories(stories: list[list[Event]]) -> dict:
    # iterate over each story
    # iterate over each event in story
    all_stories = []
    for story in stories:
        s_docs = []
        for event in story:
            s_docs.extend(d.doc_id for d in event.docs)
        all_stories.append(s_docs)
    return {"story_clusters": all_stories}


def get_or_add_keywordNode(tag: dict, graphNodes: dict) -> KeywordNode:
    if tag["baseForm"] in graphNodes:
        return graphNodes[tag["baseForm"]]
    
    keyword = Keyword(tag["baseForm"],words=tag["words"],tf=tag.get("tf", 0), df=tag.get("df", 0))
    keywordNode = KeywordNode(keyword=keyword)
    graphNodes[keyword.baseForm] = keywordNode
    return keywordNode

def update_or_create_keywordEdge(kn1: KeywordNode, kn2:KeywordNode) -> KeywordEdge:
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
   

def incremental_clustering(new_news_items: list,already_clusterd_events:list):
    # create corpus from news_items
    corpus = create_corpus(new_news_items)

    # create keyGraph from corpus
    g = KeywordGraph()
    g.build_graph(corpus=corpus)
    
    # add to g the new nodes and edges from already_clusterd_events
    for cluster in already_clusterd_events:
        tags = cluster["tags"]
        for keyword_1 in tags:
            for keyword_2 in tags:
                if keyword_1 != keyword_2:
                    keyNode1 = get_or_add_keywordNode(keyword_1, g.graphNodes)
                    keyNode2 = get_or_add_keywordNode(keyword_2, g.graphNodes)
                    # add edge and increase edge df 
                    update_or_create_keywordEdge(keyNode1,keyNode2)
                    
    # extract events
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
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


def belongs_to_story(ev, story, model) -> bool:
    text_1 = " ".join([d.title for d in ev.docs.values()])
    text_2 = " ".join([d.title for e in story for d in e.docs.values()])
    #print(text_1)
    #print(text_2)
    return compute_similarity(text_1, text_2, model) >= SimilarityThreshold
