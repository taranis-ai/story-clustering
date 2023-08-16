
def test_create_corpus():
    from story_clustering.clustering import create_corpus
    from .testdata import news_item_aggregate_1
    input = news_item_aggregate_1["news_items"]
    #print(type(input[0]))
    input[0]["tags"] = news_item_aggregate_1["tags"]
    #input.pop("tags")
    corpus = create_corpus(input)
    assert corpus.docs[13].title == "Test News Item 13"
