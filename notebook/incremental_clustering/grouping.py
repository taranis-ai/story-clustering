def group_aggregate(aggregate_ids: list[int], news_item_list):
    first_id = aggregate_ids.pop(0)
    first_aggregate = [dic for dic in news_item_list if dic['id'] == first_id][0]
    if not first_aggregate:
        print("error not_found")
    processed_aggregates = [first_aggregate]
    for item in aggregate_ids:
        aggregate = [dic for dic in news_item_list if dic['id'] == item][0]
        if not aggregate:
            continue
        first_aggregate['tags'] = first_aggregate['tags'] | aggregate['tags']
        for news_item in aggregate['news_items'][:]:
            first_aggregate['news_items'].append(news_item)
            first_aggregate['relevance'] += 1
            aggregate['news_items'].remove(news_item)
        processed_aggregates.append(aggregate)
    return first_aggregate