from story_clustering.config import Config
from story_clustering.predictor import Predictor


class PredictorFactory:
    """
    Factory class that dynamically instantiates and returns the correct Predictor
    based on the configuration. This approach ensures that only the configured model
    is loaded at startup.
    """

    def __new__(cls, *args, **kwargs) -> Predictor:
        if Config.MODEL == "tfidf":
            from story_clustering.clustering import Cluster

            return Cluster(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported NER model: {Config.MODEL}")
