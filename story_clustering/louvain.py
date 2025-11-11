from story_clustering.clustering import Cluster
from typing import Any


class Louvain:
    model_name = "louvain_method_clustering"

    def __init__(self):
        self.model = Cluster()

    def predict(self, stories: list[dict]) -> dict[str, Any]:
        if all(len(story["news_items"]) == 1 for story in stories):
            clustering_results = self.model.initial_clustering(stories)
            msg = f"Initial Clustering done with: {len(stories)} news items"
            return {"cluster_ids": clustering_results, "message": msg}

        already_clustered, to_cluster = self.model.separate_stories(stories)
        clustering_results = self.model.incremental_clustering_v2(to_cluster, already_clustered)
        msg = f"incremental Clustering done with: {len(stories)} news items"
        return {"cluster_ids": clustering_results, "message": msg}
