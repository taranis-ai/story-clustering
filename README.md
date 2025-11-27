# Taranis AI story_clustering Bot

This code takes newsitems in the format as provided by [Taranis AI](https://github.com/taranis-ai/taranis-ai) and clusters them into Stories.

## Description

The approach supports the following functionalities:

1) Automatically detect Events.
2) News items are clustered based on the detected Events.
3) Documents belonging to related Events are then clustered into Stories.

## Initial clustering

The method `initial_clustering` in `clustering.py` takes as input a dictionary of `news_items_aggregate` (see `tests/testdapa.py` for the actual input format) and outputs a dictionary containing two keys:
("event_clusters" : list of list of documents ids) and
("story_clusters" : list of list of documents ids)

## Incremental clustering

The method `incremental_clustering_v2` takes as input a dictionary of `news_items_aggregate`, containing new news items to be clustered, and `clustered_news_items_aggregate`, containing already clustered items, and tries to cluster the new documents to the existing clusters or create new ones. See `tests/testdata.py` for the actual input formats. This method also
outputs a dictionary containing two keys:
("event_clusters" : list of list of documents ids) and
("story_clusters" : list of list of documents ids)


## Pre-requisites

- uv - https://docs.astral.sh/uv/getting-started/installation/
- docker (for building container) - https://docs.docker.com/engine/

Create a python venv and install the necessary packages for the bot to run.

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras --dev
```

## Usage

You can run your bot locally with

```bash
flask run --port 5500
# or
granian app --port 5500
```

You can set configs either via a `.env` file or by setting environment variables directly.
available configs are in the `config.py`
You can select the model via the `MODEL` env var. E.g.:

```bash
MODEL=<default_model> flask run
```


## Docker

You can also create a Docker image out of this bot. For this, you first need to build the image with the build_container.sh

You can specify which model the image should be built with the MODEL environment variable. If you omit it, the image will be built with the default model.

```bash
MODEL=<model_name> ./build_container.sh
```

then you can run it with:

```bash
docker run -p 5500:8000 <image-name>:<tag>
```

If you encounter errors, make sure that port 5500 is not in use by another application.


## Test the bot

Once the bot is running, you can send test data to it on which it runs its inference method:

```bash
curl -X POST http://127.0.0.1:5500 -H "Content-Type: application/json" -d '{"key": "some data"}'
```

## Development

If you want to contribute to the development of this bot, make sure you set up your pre-commit hooks correctly:

- Install pre-commit (https://pre-commit.com/)
- Setup hooks: `> pre-commit install`


## License

EUROPEAN UNION PUBLIC LICENCE v. 1.2
