from flask import Flask, Blueprint, jsonify, request
from flask.views import MethodView

from story_clustering.clustering import Cluster
from story_clustering.log import logger
from story_clustering.nlp_utils import separate_stories


class Clustering(MethodView):
    def post(self, cluster: Cluster):
        data = request.get_json()
        stories = data.get("stories", [])

        if all(len(story["news_items"]) == 1 for story in stories):
            clustering_results = cluster.initial_clustering(stories)
            logger.info(f"Initial Clustering done with: {len(stories)} news items")
            return jsonify({"cluster_ids": clustering_results})

        already_clustered, to_cluster = separate_stories(stories)
        clustering_results = cluster.incremental_clustering_v2(to_cluster, already_clustered)
        logger.info(f"incremental Clustering done with: {len(stories)} news items")
        return jsonify({"cluster_ids": clustering_results})


class HealthCheck(MethodView):
    def get(self):
        return jsonify({"status": "ok"})


def init(app: Flask, cluster: Cluster):
    app.url_map.strict_slashes = False

    clustering_bp = Blueprint("predict", __name__)
    clustering_bp.add_url_rule("/cluster", view_func=Clustering.as_view("cluster", cluster=cluster))
    clustering_bp.add_url_rule("/health", view_func=HealthCheck.as_view("health"))
    app.register_blueprint(clustering_bp)
