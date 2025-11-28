import numpy as np
from typing import Any


class Gte:
    model_name = "embedding_based_clustering"

    def __init__(self):
        pass

    def compute_centroid(self, stories_in_cluster: list[dict]) -> np.ndarray:
        """Compute normalized centroid from all news_items of all stories."""
        all_embeddings = []
        for story in stories_in_cluster:
            all_embeddings.extend(news_item["embedding"] for news_item in story["news_items"])

        centroid = np.mean(np.vstack(all_embeddings), axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def predict(self, stories: list[dict]) -> dict[str, Any]:
        msg = "ok"
        threshold = 0.8

        clusters = [[s] for s in stories]

        while True:
            centroids = [self.compute_centroid(cluster) for cluster in clusters]

            embedding_matrix = np.vstack(centroids)
            sim_matrix = embedding_matrix @ embedding_matrix.T
            np.fill_diagonal(sim_matrix, -np.inf)

            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            max_sim = sim_matrix[i, j]

            if max_sim < threshold:
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
