import numpy as np
from typing import Any
from story_clustering.structure_compare import StructureTokenizer
from itertools import combinations
import math


class Gte:
    model_name = "embedding_based_clustering"

    def __init__(self):
        self.cmp = StructureTokenizer()

    def compute_centroid(self, stories_in_cluster: list[dict]) -> np.ndarray:
        """Compute normalized centroid from all news_items of all stories."""
        all_embeddings = []
        for story in stories_in_cluster:
            all_embeddings.extend(news_item["embedding"] for news_item in story["news_items"])

        centroid = np.mean(np.vstack(all_embeddings), axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def compute_layout_similarity(self, cluster: list[dict]) -> float:
        """Compute the layout similarity pairwise between all news items in cluster1 and cluster2"""
        layout_sim = 0.0
        news_items = []
        for story in cluster:
            news_items.extend(story["news_items"])

        for news_item_pair in combinations(news_items, 2):
            t1 = self.cmp.tokenize(news_item_pair[0]["content"])
            t2 = self.cmp.tokenize(news_item_pair[1]["content"])
            layout_sim += self.cmp.calc_fuzzy_similarity(t1, t2)
        return layout_sim / math.comb(len(news_items), 2)

    def predict(self, stories: list[dict]) -> dict[str, Any]:
        msg = "ok"
        threshold = 0.75

        clusters = [[s] for s in stories]

        while True:
            centroids = [self.compute_centroid(cluster) for cluster in clusters]

            embedding_matrix = np.vstack(centroids)
            sim_matrix = embedding_matrix @ embedding_matrix.T
            np.fill_diagonal(sim_matrix, -np.inf)

            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            max_sim = sim_matrix[i, j]

            # take into account similar layout
            layout_sim = self.compute_layout_similarity(clusters[i] + clusters[j])
            total_sim = max_sim * np.exp(-layout_sim / 4)

            if total_sim < threshold:
                break

            clusters[i].extend(clusters[j])
            clusters[j] = []

            clusters = [c for c in clusters if len(c) > 0]
            if len(clusters) <= 1:
                break

        event_clusters = []
        for cluster in clusters:
            if cluster:
                event_clusters.append([story["id"] for story in cluster])

        clustering_results = {"event_clusters": event_clusters}
        return {"cluster_ids": clustering_results, "message": msg}
