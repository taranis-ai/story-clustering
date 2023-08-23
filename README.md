# Story Clustering

This code takes newsitems in the format as provided by [Taranis-NG](https://github) and clusters them into Stories.


# Description and Use

The approach supports the following functionalities:
1) Automatically detect Events.
2) News items are clustered based on the detected Events.
3) Documents belonging to related Events are then clustered into Stories.

## Initial clustering

The method `initial_clustering` in `clustering.py` takes as input a dictionary of `news_items_aggregate` (see `tests/testdapa.py` for the actual input format) and outputs a dictionary containing two keys:
("event_clusters" : list of list of documents ids) and 
("story_clusters" : list of list of documents ids) 

## Incremental clustering
The incremental clustering method takes as input a dictionary of `news_items_aggregate`, containing new news items to be clustered, and `clustered_news_items_aggregate`, containing already clustered items, and tries to cluster the new documents to the existing clusters or create new ones. See `tests/testdata.py` for the actual input formats. This method also 
outputs a dictionary containing two keys:
("event_clusters" : list of list of documents ids) and 
("story_clusters" : list of list of documents ids) 

## Installation
The `requirements.txt` file should list all Python libraries that the story-clustering
depends on, and they will be installed using:

```
pip install .
```

## Development
```
pip install .[dev]
```

## Use
See `notebook\test_story_clustering.ipynb` for examples on how to use the clustering methods.



## License
EUROPEAN UNION PUBLIC LICENCE v. 1.2



