import torch
import nltk
import logging
from sentence_transformers import util

from story_clustering.predictor import Predictor
from story_clustering.document_representation import Keyword, Document, Corpus
from story_clustering.event_organizer import Event
from story_clustering.eventdetector import (
    extract_events_from_corpus,
    extract_topic_by_keyword_communities,
    calc_docs_tfidf_vector_size_with_graph_2,
    compute_similarity,
    SimilarityThreshold,
)
from story_clustering.keywords_organizer import KeywordGraph, KeywordEdge, KeywordNode
from story_clustering.nlp_utils import (
    compute_tf,
    replace_umlauts_with_digraphs,
    find_keywords_matches,
    get_sentence_transformer,
    get_stop_words,
    POLYFUZZ_THRESHOLD,
)

from story_clustering.log import logger


HIGH_PRIORITY = 10
MID_HIGH_PRIORITY = 8
MID_PRIORITY = 5
MID_LOW_PRIORITY = 3
LOW_PRIORITY = 1


class Cluster(Predictor):
    def __init__(self):
        nltk.download("stopwords")

        bert_logger = logging.getLogger("BERTopic")

        self.stopwords = get_stop_words()

        class IgnoreSpecificLogFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "Ran model with model id" not in record.getMessage()

        bert_logger.addFilter(IgnoreSpecificLogFilter())

        self.sentence_transformer = get_sentence_transformer()

        logger.info("Story Clustering Setup Complete.")

    def create_corpus(self, new_stories: list[dict]) -> Corpus:
        """Creates a Corpus object from a JSON object denoting all documents

        Args:
            new_stories (list[dict]): list of dict with following keys
                {'id': str, 'link': str, 'text': str, 'title':str,'date': 'YYYY-MM-DD', 'lang': str,
                'tags': {"<tag_name>": {"name": "<tag_name>", "tag_type": "<tag_type>"}, ...}
        Returns:
            corpus: Corpus of documents
        """
        corpus = Corpus()
        for story in new_stories:
            doc = Document(doc_id=story["id"])
            for nitem in story["news_items"]:
                doc.url = nitem.get("link", None)
                doc.content = nitem["content"] or nitem["review"]
                if not doc.content:
                    continue
                doc.title = nitem["title"]
                if doc.title is not None:
                    doc.segTitle = doc.title.strip().split(" ")
                doc.publish_time = nitem.get(".published", None)
                doc.language = nitem["language"]
                keywords = {}
                if len(story["tags"]) < 5:
                    continue
                for tag, tag_dict in story["tags"].items():
                    tag_type = tag_dict.get("tag_type", "")
                    if (tag not in doc.content) and (tag.lower() not in doc.content):
                        continue
                    baseform = replace_umlauts_with_digraphs(tag)
                    keyword = Keyword(
                        baseForm=baseform,
                        tf=0,
                        df=0,
                        documents=set(),
                    )
                    keywords[baseform] = keyword

                    keyword.tf = self.compute_tf_with_boost(baseform, doc.content, tag_type=tag_type)

                doc.keywords = keywords
                corpus.docs[doc.doc_id] = doc

        corpus.update_df()
        logger.debug(f"Corpus size: {len(corpus.docs)}")
        return corpus

    def compute_tf_with_boost(self, baseform: str, content: str, tag_type: str) -> int:
        tf = 0
        if tag_type in {"APT", "cves"}:
            tf = HIGH_PRIORITY
        elif tag_type in {
            "Company",
            "sha256s",
            "sha1s",
            "registry_key_paths",
            "md5s",
            "bitcoin_addresses",
        }:
            tf = MID_HIGH_PRIORITY
        elif tag_type in {"Country", "CVE_VENDOR"}:
            tf = MID_PRIORITY
        elif tag_type in {"PER", "LOC", "ORG"}:
            tf = MID_LOW_PRIORITY
        else:
            tf = LOW_PRIORITY

        tf += compute_tf(baseform, content)
        return tf

    def initial_clustering(self, stories: list):
        corpus = self.create_corpus(stories)

        events = extract_events_from_corpus(corpus=corpus)
        return self.to_json_events(events)

    def compute_df(self, keyword_baseform: str, cluster_news_items: list) -> int:
        df = 0
        for nitem in cluster_news_items:
            content = nitem["content"] or nitem["review"]
            if not content:
                continue
            if keyword_baseform.lower() in content:
                df += 1
        return df if df > 0 else 1

    def create_keygraph(self, cluster: dict, corpus: Corpus) -> KeywordGraph:
        graph = KeywordGraph(story_id=cluster["id"])
        # as text we use for now the content of first item
        graph.text = cluster["news_items"][0]["content"]
        tag_names = list(cluster.get("tags", {}).keys())

        # update corpus DF for each of the tags
        # use corpus.DF[baseform] to update the df of each keyword
        for keyword_1 in tag_names:
            baseform1 = replace_umlauts_with_digraphs(keyword_1)
            if baseform1 in corpus.DF:
                corpus.DF[baseform1] += self.compute_df(keyword_1, cluster["news_items"])
            else:
                corpus.DF[baseform1] = self.compute_df(keyword_1, cluster["news_items"])

        for keyword_1 in tag_names:
            baseform1 = replace_umlauts_with_digraphs(keyword_1)
            for keyword_2 in tag_names:
                if keyword_1 != keyword_2:
                    baseform2 = replace_umlauts_with_digraphs(keyword_2)

                    keyNode1 = self.get_or_add_keywordNode(keyword_1, graph.graphNodes, corpus.DF[baseform1])
                    keyNode2 = self.get_or_add_keywordNode(keyword_2, graph.graphNodes, corpus.DF[baseform2])
                    # add edge and increase edge df
                    self.update_or_create_keywordEdge(keyNode1, keyNode2)

        return graph

    def create_communities_incr_clustering(self, corpus: Corpus, already_clustered_events: list):
        corpus.update_df()
        docs_size = len(corpus.docs)

        # create keygraph for each cluster
        existing_communities = []
        for cluster in already_clustered_events:
            existing_communities.append(self.create_keygraph(cluster, corpus))
            docs_size += len(cluster["news_items"])

        calc_docs_tfidf_vector_size_with_graph_2(corpus.docs, corpus.DF, existing_communities)

        return existing_communities, docs_size

    def merge_cluster(self, new_cluster: Event, already_clustered_events):
        super_cluster_match = -1
        # use polyfuzz to find the intersection
        keywords_new_cluster = list(new_cluster.keyGraph.graphNodes.keys())
        super_cluster_id = None
        for cluster in already_clustered_events:
            existing_keywords = []
            tag_names = list(cluster.get("tags", {}).keys())
            for tag in tag_names:
                baseform = replace_umlauts_with_digraphs(tag)
                existing_keywords.append(baseform)
            if find_keywords_matches(keywords_new_cluster, existing_keywords) > POLYFUZZ_THRESHOLD:
                new_cluster_content = f"{list(new_cluster.docs.values())[0].title} {list(new_cluster.docs.values())[0].content}"
                existing_cluster_content = cluster["title"] + cluster["description"]
                if len(cluster["news_items"]) > 0:
                    existing_cluster_content = cluster["news_items"][0]["content"] or cluster["news_items"][0]["review"]
                matching_score = compute_similarity(new_cluster_content, existing_cluster_content)
                if matching_score >= SimilarityThreshold and matching_score > super_cluster_match:
                    super_cluster_match = matching_score
                    super_cluster_id = cluster["id"]

        return super_cluster_id

    def incremental_clustering_v3(self, new_news_items: list, already_clustered_events: list):
        corpus = self.create_corpus(new_news_items)
        events = extract_events_from_corpus(corpus=corpus)

        for ev in events:
            super_cluster_id = self.merge_cluster(ev, already_clustered_events)
            if super_cluster_id is not None:
                ev.keyGraph.story_id = super_cluster_id
                print(f"Merged to cluster: {super_cluster_id}")

        return self.to_json_events(events)

    def incremental_clustering_v2(self, new_news_items: list, already_clustered_events: list):
        corpus = self.create_corpus(new_news_items)

        existing_communities, docs_size = self.create_communities_incr_clustering(corpus, already_clustered_events)

        updated_events = extract_topic_by_keyword_communities(corpus, existing_communities, docs_size)
        return self.to_json_events(updated_events)

    @DeprecationWarning
    def incremental_clustering(self, new_news_items: list, already_clustered_events: list):
        corpus = self.create_corpus(new_news_items)

        # create keyGraph from corpus
        graph = KeywordGraph()
        graph.build_graph(corpus=corpus)

        # add to g the new nodes and edges from already_clusterd_events
        for cluster in already_clustered_events:
            tag_names = list(cluster.get("tags", {}).keys())
            for keyword_1 in tag_names:
                for keyword_2 in tag_names:
                    if keyword_1 != keyword_2:
                        # doc frequency is the number of documents in the cluster
                        df = len(cluster["news_items"])
                        keyNode1 = self.get_or_add_keywordNode(keyword_1, graph.graphNodes, df)
                        keyNode2 = self.get_or_add_keywordNode(keyword_2, graph.graphNodes, df)
                        # add edge and increase edge df
                        self.update_or_create_keywordEdge(keyNode1, keyNode2)

        events = extract_events_from_corpus(corpus=corpus, graph=graph)
        return self.to_json_events(events)

    def cluster_stories_from_events(self, events: list[Event]) -> list[list[Event]]:
        stories = []
        for event in events:
            found_story = False
            for story in stories:
                if self.belongs_to_story(event, story):
                    story.append(event)
                    found_story = True
                    break
            if not found_story:
                aux = [event]
                stories.append(aux)
        return stories

    def to_json_events(self, events: list[Event]) -> dict:
        all_events = []
        for event in events:
            docs_in_event = []
            if event.keyGraph.story_id:
                if event.keyGraph.story_id not in docs_in_event:
                    docs_in_event.append(event.keyGraph.story_id)
            if event.docs:
                docs_in_event.extend(list(event.docs.keys()))

            if len(docs_in_event) > 0:
                all_events.append(docs_in_event)
        # all_events = [list(event.docs.keys()) for event in events if event.docs]

        # keywords = [event.keyGraph.graphNodes.keys() for event in events if event.docs]
        return {"event_clusters": all_events}  # , "events_keywords":keywords}

    def to_json_stories(self, stories: list[list[Event]]) -> dict:
        all_stories = []
        for story in stories:
            s_docs = []
            for event in story:
                s_docs.extend(list(event.docs))
            all_stories.append(s_docs)
        return {"story_clusters": all_stories}

    def get_or_add_keywordNode(self, tag: str, graphNodes: dict, df: int) -> KeywordNode:
        baseform = replace_umlauts_with_digraphs(tag)
        if baseform in graphNodes:
            node = graphNodes[baseform]
            # node.keyword.increase_df(df)
            node.keyword.df = df
            return node

        keyword = Keyword(baseForm=baseform, tf=0, df=df)
        keywordNode = KeywordNode(keyword=keyword)
        graphNodes[keyword.baseForm] = keywordNode
        return keywordNode

    def update_or_create_keywordEdge(self, kn1: KeywordNode, kn2: KeywordNode):
        edgeId = KeywordEdge.get_id(kn1, kn2)
        if edgeId not in kn1.edges:
            new_edge = KeywordEdge(kn1, kn2, edgeId)
            new_edge.df += 1
            kn1.edges[edgeId] = new_edge
            kn2.edges[edgeId] = new_edge
        else:
            kn1.edges[edgeId].df += 1

            if kn1.edges[edgeId].df != kn2.edges[edgeId].df:
                kn2.edges[edgeId].df = kn1.edges[edgeId].df

    def compute_similarity_for_stories(self, text_1, text_2):
        sent_text_1 = text_1.replace("\n", " ").split(".")
        sent_text_2 = text_2.replace("\n", " ").split(".")

        sent_text_2 = [s for s in sent_text_2 if s != ""]
        sent_text_1 = [s for s in sent_text_1 if s != ""]

        if not sent_text_1 or not sent_text_2:
            return 0

        em_1 = self.sentence_transformer.encode(sent_text_1, convert_to_tensor=True, show_progress_bar=False)
        em_2 = self.sentence_transformer.encode(sent_text_2, convert_to_tensor=True, show_progress_bar=False)

        consine_sim_1 = util.pytorch_cos_sim(em_1, em_2)
        max_vals, _inx = torch.max(consine_sim_1, dim=1)
        avg = torch.mean(max_vals, dim=0)
        return avg.item()

    def belongs_to_story(self, ev, story) -> bool:
        text_1 = " ".join([d.title for d in ev.docs.values()])
        text_2 = " ".join([d.title for e in story for d in e.docs.values()])
        return self.compute_similarity_for_stories(text_1, text_2) >= SimilarityThreshold

    def predict(self, text: str) -> dict[str, str]:
        return {"NOT_IMPLEMENTED": "Not implemented yet."}
