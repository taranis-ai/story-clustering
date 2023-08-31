def test_create_corpus():
    from story_clustering.clustering import create_corpus
    from .testdata import news_item_list

    corpus = create_corpus(news_item_list)
    assert corpus.docs[13].title == "Test News Item 13"
    assert "software" in corpus.docs[27].keywords.keys()


def test_initial_clustering():
    from story_clustering.clustering import initial_clustering
    from .testdata import news_item_list

    clustering_results = initial_clustering(news_item_list)
    print(clustering_results)
    assert clustering_results == {"event_clusters": [[13], [27, 23], [93], [1414, 12, 1337], [11], [4242], [137]]}


def test_incremental_clustering():
    from story_clustering.clustering import incremental_clustering
    from .testdata import news_item_list, clustered_news_item_list

    clustering_results = incremental_clustering(news_item_list, clustered_news_item_list)
    print(clustering_results)
    assert clustering_results == {"event_clusters": [[13], [27, 23], [93], [1414, 12, 1337], [11], [4242], [137]]}
