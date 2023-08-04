from .document_representation import Keyword, Document, Corpus
from sentence_transformers import SentenceTransformer, util
from .event_organizer import Event
from .eventdetector import extractEventsFromCorpus

import math
from .nlp_utils import tfidf, idf
import logging
import torch
import json
import time

EventSplitAlg = 'DocGraph'
MinTopicSize = 1


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
        doc = Document()
        doc.doc_id = nitem['id']
        
        doc.url = nitem.get('url', None)
        doc.content = nitem['text']
        doc.title = nitem['title']
        doc.publishTime = nitem.get('date', None)
        doc.language = nitem['lang']
        # create keywords
        keywords = dict() 
        for tag in nitem['tags']:
            keyword = Keyword(baseform=tag['baseform'], words=tag['words'],tf=tag.get('tf',0),df=tag.get('df',0))
            keywords[tag['baseForm']] = keyword
            
            # TODO: check if tf and df are computed and do so if not
            
        doc.keywords = keywords
        corpus.docs[doc.doc_id] = doc
    
    corpus.updateDF()
    return corpus


def initial_clustering(new_news_items: list):
    # create corpus
    corpus = create_corpus(new_news_items)
    
    # extract events
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    events = extractEventsFromCorpus(corpus=corpus, model=model)
    
    # create stories based on events
    stories = list()
    for e in events:
        found_story = False
        for s in stories:
            if belongsToStory(e,s,model):
                s.append(e)
                found_story = True
                break
        if not found_story:
            aux = list()
            aux.append(e)
            stories.append(aux)
    
    
    new_aggregates = to_json_events(events)
    new_aggregates.append(to_json_stories(stories))
    return new_aggregates


def to_json_events(events:list[Event]) -> str:
    # iterate over each event and return the list of documents ids belonging to the same event
    all_events = list()
    for e in events:
        e_docs = list()
        for d in e.docs:
            e_docs.append(d.doc_id)
        all_events.append(e)
    return {'event_clusters': all_events}
            
        
        


def to_json_stories(stories:list[list[Event]]) -> str:
    # iterate over each story s 
    # iterate over each event in s 
    all_stories = list()
    for s in stories:
        s_docs = list()
        for e in s:
            for d in e.docs:
                s_docs.append(d.doc_id)
        all_stories.append(s_docs)
    return {'story_clusters': all_stories}
            
            
    

def incremental_clustering(new_news_items: list, already_clustered_events:list):
    # create corpus
    corpus = Corpus()
    for nitem in new_news_items:
        doc = Document()
        doc.doc_id = nitem['id']
        doc.url = nitem['url']
        doc.content = nitem['text']
        doc.title = nitem['title']
        doc.publishTime = nitem['date']
        doc.language = nitem['lang']
        # create keywords
        keywords = dict() 
        for tag in nitem['tags']:
            keyword = Keyword(baseform=tag['baseform'], words=tag['words'],tf=tag['tf'],df=tag['df'])
            keywords[tag['baseForm']] = keyword
        doc.keywords = keywords
        corpus.docs[doc.doc_id] = doc
    
    corpus.updateDF()
    
    # create KeyGraph from existing event_clusters
    
    
    
    
    
    
    
    


    


def compute_similarity(text_1, text_2,model):
    sent_text_1 = text_1.replace('\n',' ').split('.')
    sent_text_2 = text_2.replace('\n',' ').split('.')
    
    sent_text_2 = [s for s in sent_text_2 if s != ''][:10]
    sent_text_1 = [s for s in sent_text_1 if s != ''][:10]
    
    em_1 = model.encode(sent_text_1, convert_to_tensor=True)
    em_2 = model.encode(sent_text_2, convert_to_tensor=True)
    
    consine_sim_1 = util.pytorch_cos_sim(em_1,em_2)
    max_vals, _inx = torch.max(consine_sim_1, dim=1)
    avg = torch.mean(max_vals, dim=0)
    return avg.item()

def belongsToStory(ev,story,model) -> bool:
    text_1 = " ".join([d.title for d in ev.docs.values()])
    text_2 = " ".join([d.title for e in story for d in e.docs.values()])
    print(text_1)
    print(text_2)
    if compute_similarity(text_1,text_2,model) >= 0.44:
        return True
    return False