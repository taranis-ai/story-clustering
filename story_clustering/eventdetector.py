from .keywords_organizer import KeywordGraph, CommunityDetector, KeywordNode, KeywordEdge
from sentence_transformers import SentenceTransformer, util
from .document_representation import Keyword, Document, Corpus
from .event_organizer import Event
import math
from .nlp_utils import tfidf, idf
#import logging
import torch
import json
#import time

EventSplitAlg = 'DocGraph'
MinTopicSize = 1



def extractEventsFromCorpus(corpus: Corpus, model) -> list[Event]: 
    g = KeywordGraph()
    g.buildGraph(corpus)
    
    # extract keyword communities from keyword graph
    calcDocsTFIDFVectorSizeWithGraph(corpus.docs, corpus.DF, g.graphNodes)
    
    communities = CommunityDetector(g.graphNodes).detectCommunitiesLouvain()
    s = len(communities)
    print(f'Communities size: {s}')
    
    # extract events from corpus based on keyword communities
    events = extractTopicByKeywordCommunities(corpus, communities)
    print('docs to events assigned. Detecting sub-events...')
    
    # identify more fine-grained events
    events = splitEvents(events,corpus.DF,len(corpus.docs),model)
                         
    for e in events:
        e.refineKeyGraph()
    
    return events

def calcDocsTFIDFVectorSizeWithGraph(docs: dict[str,Document], DF: dict[str,float], graphNodes: dict[str,KeywordNode]):
    for d in docs.values():
        d.tfidfVectorSizeWithKeygraph = 0
        for k in d.keywords.values():
            if k.baseForm in graphNodes:
                d.tfidfVectorSizeWithKeygraph += math.pow(tfidf(k.tf, idf(DF[k.baseForm],len(docs))),2)
        
        d.tfidfVectorSizeWithKeygraph = math.sqrt(d.tfidfVectorSizeWithKeygraph)

def extractTopicByKeywordCommunities(corpus: Corpus, communities: dict[str,KeywordNode]) -> list[Event]:
    result = list()
    doc_community = dict()
    doc_similarity = dict()
    # initialization
    for d in corpus.docs.values():
        doc_community[d.doc_id] = -1
        doc_similarity[d.doc_id] = -1.0
    
    
    for i,c in enumerate(communities):
        for doc in corpus.docs.values():
            cosineSimilarity = tfidfCosineSimilarityGraph2Doc(c, doc, corpus.DF, len(corpus.docs))
            if cosineSimilarity > doc_similarity[doc.doc_id]:
                doc_community[doc.doc_id] = i
                doc_similarity[doc.doc_id] = cosineSimilarity
    
    
    # create event for each community
    for i,c in enumerate(communities):
        e = Event()
        e.keyGraph = c
        
        for doc_id in doc_community:
            if doc_community[doc_id] == i:
                d = corpus.docs[doc_id]
                if d.doc_id not in e.docs:
                    e.docs[d.doc_id] = d
                    e.similarities[d.doc_id] = doc_similarity[doc_id]
                d.processed =  True
        
        if len(e.docs) >= MinTopicSize:
            result.append(e)
    return result
            
def tfidfCosineSimilarityGraph2Doc(community: dict[str,KeywordNode], d2: Document, DF: dict[str,float], docSize:int) -> float:
    sim = 0
    vectorsize1 = 0
    numberOfKeywordsInCommon = 0
    
    for n in community.graphNodes.values():
        # calculate community keyword's tf
        nTF = 0
        for e in n.edges.values():
            e.computeCPs()
            nTF += max(e.cp1,e.cp2)
        n.keyword.tf = nTF /len(n.edges)
    
        if n.keyword.baseForm in DF:
            # update vector size of community
            vectorsize1 += math.pow(tfidf(n.keyword.tf, idf(DF[n.keyword.baseForm],docSize)),2)
        
            # update similarity between document d2 and community
            if n.keyword.baseForm in d2.keywords:
                numberOfKeywordsInCommon += 1
                sim += tfidf(n.keyword.tf, idf(DF[n.keyword.baseForm], docSize)) * \
                    tfidf(d2.keywords[n.keyword.baseForm].tf, idf(DF[n.keyword.baseForm], docSize))
    vectorsize1 = math.sqrt(vectorsize1)
    
    # return similarity
    if vectorsize1 > 0 and d2.tfidfVectorSizeWithKeygraph > 0:
        return sim / vectorsize1 / d2.tfidfVectorSizeWithKeygraph
    else:
        return 0

            

def splitEvents(events: list[Event], DF: dict[str,float], docAmount:int, model) -> list[Event]:
    # updated the original implementation to use sentence transformers to detect if
    # two documents talk about the same event
    result = list()
    for e in events:
        if len(e.docs) >= 2:
            split_events = list()
            docKeys = [d for d in e.docs.keys()]
            processed_doc_keys = list()
            
            for i, d1 in enumerate(e.docs.keys()):
                if d1 in processed_doc_keys:
                    continue
                else:
                    processed_doc_keys.append(d1)
                    sub_event = Event()
                    sub_event.keyGraph = KeywordGraph()
                    sub_event.keyGraph.graphNodes = e.keyGraph.graphNodes.copy()
                    sub_event.docs[d1] = e.docs[d1]
                    sub_event.similarities[d1] = e.similarities[d1]
                    
                    for j in range(i+1,len(e.docs.keys())):
                        d2 = list(e.docs.keys())[j]
                        if d2 in processed_doc_keys:
                            continue
                        else:
                            is_same_event = sameEvent(e.docs[d1],e.docs[d2],DF,docAmount,model)
                            if not is_same_event:
                                continue
                            else:
                                sub_event.docs[d2] = e.docs[d2]
                                sub_event.similarities[d2] = e.similarities[d2]
                                processed_doc_keys.append(d2)
                    
                    split_events.append(sub_event)
            result.extend(split_events)
        else:
            result.append(e)
    return result

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

def sameEvent(d1:Document,d2:Document,DF,docAmount,model) -> bool:
    text_1 = d1.content
    text_2 = d2.content
    if compute_similarity(text_1,text_2,model) >= 0.44:
        return True
    return False
                                
    

def create_corpus_from_json(corpus_dict) -> Corpus: 
    docs = dict()
    DF = dict()                                              
    corpus = Corpus()
    #print(type(corpus_dict['docs']))
    
    #list_ids = [0,4,29,30]
    for doc_id,doc in corpus_dict['docs'].items():
        #if int(doc_id) not in list_ids:
        #    continue
        keywords = dict()
        
        for k in doc['keywords']:
            #print(type(k))
            keyword = Keyword(baseform=k['baseForm'],
                            words=k['words'], tf=k['tf'],df=k['df'])
            keywords[k['baseForm']] = keyword
        corpus.docs[doc_id] = Document(doc_id=doc_id,url=doc['url'],publish_date=doc['publish_time'],language=doc['language'],
                                title=doc['title'],content=doc['content'],keywords=keywords)
    # TODO: uncomment after debugging
    #for word, tf in corpus_dict['DF'].items():
    #    DF[word] = tf
    #corpus.DF = DF
    
    return corpus
                    
 


    
    
            
    
                   



if __name__ == "__main__":
    # read and create Corpus
    f = open('awake/data/corpus_test.json','r')
    #print(corpus)
    corpus_dict = json.load(f)
    f.close()
    print('Corpus loaded...')
    corpus = create_corpus_from_json(corpus_dict)
    corpus.updateDF()
    
    print('Corpus created...')
    # extract events from corpus
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    events = extractEventsFromCorpus(corpus=corpus,model=model)
    for e in events:
        print('Titles:')
        for d in e.docs:
            print(e.docs[d].title)
        print('------')
        print(f'Keywords: {list(e.keyGraph.graphNodes.keys())}')
        print('-------------')
        
            
        
    
        
        