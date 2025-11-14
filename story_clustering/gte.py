import numpy as np
from typing import Any


class Gte:
    model_name = "embedding_based_clustering"

    def __init__(self):
        pass

    def _compute_centroid(self, stories_in_cluster):
        """Compute normalized centroid from all news_items of all stories."""
        all_embeddings = []
        for story in stories_in_cluster:
            for ni in story["news_items"]:
                all_embeddings.append(np.array(ni["embedding"], dtype=float))

        centroid = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def predict(self, stories: list[dict]) -> dict[str, Any]:
        msg = "ok"
        threshold = 0.8

        # Start with each story as its own cluster
        clusters = [[s] for s in stories]

        while True:
            # ---- Recompute centroids for all clusters ----
            centroids = []
            for cluster in clusters:
                if len(cluster) == 0:
                    centroids.append(None)
                else:
                    centroids.append(self._compute_centroid(cluster))

            # ---- Build similarity matrix ----
            active_centroids = [c for c in centroids if c is not None]
            if len(active_centroids) <= 1:
                break  # nothing to merge

            embedding_matrix = np.vstack(active_centroids)
            sim_matrix = embedding_matrix @ embedding_matrix.T

            # prevent self-matching
            np.fill_diagonal(sim_matrix, -np.inf)

            # ---- Find the most similar cluster pair ----
            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            max_sim = sim_matrix[i, j]

            if max_sim < threshold:
                break  # stop merging

            # ---- Merge clusters i and j ----
            clusters[i].extend(clusters[j])
            clusters[j] = []  # mark as deleted

        # ---- Clean empty clusters and extract story_ids ----
        event_clusters = []
        for cluster in clusters:
            if cluster:
                event_clusters.append([story["id"] for story in cluster])

        clustering_results = {"event_clusters": event_clusters}
        return {"cluster_ids": clustering_results, "message": msg}
