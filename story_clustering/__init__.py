from flask import Flask
from story_clustering import router, clustering


def create_app():
    app = Flask(__name__)
    app.config.from_object("story_clustering.config.Config")

    with app.app_context():
        init(app)

    return app


def init(app: Flask):
    cluster = clustering.Cluster()
    router.init(app, cluster)


if __name__ == "__main__":
    create_app()
