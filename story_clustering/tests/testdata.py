import copy

base_aggregate = {
    "title": "Test Aggregate",
    "description": "Test Aggregate",
    "created": "2023-08-01T17:01:04.801870",
    "read": False,
    "important": False,
    "likes": 0,
    "dislikes": 0,
    "relevance": 0,
    "comments": "",
    "summary": "",
    "news_items": [
        {
            "news_item_data": {
                "review": "",
                "author": "",
                "source": "https://url/",
                "link": "https://url/",
                "language": None,
                "osint_source_id": "78049551-dcef-45bd-a5cd-4fe842c4d5e2",
            },
        }
    ],
    "tags": {},
}


news_item_aggregate_1 = {
    "id": 1,
    "news_items": [
        {
            "id": 13,
            "news_item_data_id": "4b9a5a9e-04d7-41fc-928f-99e5ad608ebb",
            "news_item_aggregate_id": 1,
            "news_item_data": {
                "id": "4b9a5a9e-04d7-41fc-928f-99e5ad608ebb",
                "hash": "a96e88baaff421165e90ac4bb9059971b86f88d5c2abba36d78a1264fb8e9c87",
                "title": "Test News Item 13",
                "content": "CVE-2020-1234 - Test Aggregate 1, Securities",
                "collected": "2023-08-01T17:01:04.802015",
                "published": "2023-08-01T17:01:04.801998",
                "updated": "2023-08-01T17:00:39.893435",
            },
        }
    ],
    "tags": {
        "CVE-2020-1234": {"name": "CVE-2020-1234", "tag_type": "CVE", "sub_forms": ["CVE"]},
        "Security": {"name": "Security", "tag_type": "MISC", "sub_forms": ["securities"]},
    },
}


news_item_aggregate_2 = {
    "id": 2,
    "news_items": [
        {
            "id": 27,
            "news_item_data_id": "4b9a5a9e-04d7-41fc-928f-99e5ad608ebb",
            "news_item_aggregate_id": 2,
            "news_item_data": {
                "id": "4b9a5a9e-04d7-41fc-928f-99e5ad608ebb",
                "hash": "a96e88baaff421165e90ac4bb9059971b86f88d5c2abba36d78a1264fb8e9c87",
                "title": "Test News Item 27",
                "content": "CVE-2020-4321 - Test Aggregate 2, Software",
                "collected": "2023-08-01T17:01:04.802015",
                "published": "2023-08-01T17:01:04.801998",
                "updated": "2023-08-01T17:00:39.893435",
            },
        }
    ],
    "tags": {
        "CVE-2020-4321": {"name": "CVE-2020-4321", "tag_type": "CVE", "sub_forms": ["CVE"]},
        "Software": {"name": "Software", "tag_type": "MISC", "sub_forms": ["softwares"]},
    },
}

news_item_aggregate_3 = {
    "id": 3,
    "news_items": [
        {
            "id": 93,
            "news_item_data_id": "533086da-c8c1-4f8e-b3ee-103268983580",
            "news_item_aggregate_id": 3,
            "news_item_data": {
                "id": "533086da-c8c1-4f8e-b3ee-103268983580",
                "hash": "f4c7b52ecfe6ab612db30e7fa534b470fd11493fc92f30575577b356b2a1abc7",
                "title": "Test News Item",
                "content": "Long and random text - bla bla foo bar lorem ipsum",
                "collected": "2023-08-01T17:01:04.801951",
                "published": "2023-08-01T17:01:04.801934",
                "updated": "2023-08-01T17:00:39.893435",
            },
        }
    ],
    "tags": {},
}

news_item_aggregate_4 = {
    "id": 4,
    "news_items": [
        {
            "id": 1414,
            "news_item_data_id": "f8912dab-h345-6789-01jk-5lmn6789o012",
            "news_item_aggregate_id": 4,
            "news_item_data": {
                "id": "f8912dab-h345-6789-01jk-5lmn6789o012",
                "hash": "d250dgfg45gh09824j59h56jk7k82l0m53o4p91q5rst9u809v567wx8910y234",
                "title": "Test News Item",
                "content": "Breaking: Local tech wizard claims to have found the solution to the world's biggest cybersecurity threat – turning it off and then back on again! Industries are flabbergasted by the simplicity of the remedy.",
                "collected": "2023-08-05T23:11:04.801886",
                "published": "2023-08-05T23:11:04.801870",
                "updated": "2023-08-05T23:10:39.893435",
            },
        }
    ],
    "tags": {
        "tech wizard": {"name": "tech wizard", "tag_type": "MISC", "sub_forms": []},
        "cybersecurity": {"name": "cybersecurity", "tag_type": "MISC", "sub_forms": []},
        "remedy": {"name": "remedy", "tag_type": "MISC", "sub_forms": []},
    },
}


news_item_aggregate_5 = {
    "id": 5,
    "news_items": [
        {
            "id": 11,
            "news_item_data_id": "c12a3bde-a333-4567-90ab-2ed123f45678",
            "news_item_aggregate_id": 5,
            "news_item_data": {
                "id": "c12a3bde-a333-4567-90ab-2ed123f45678",
                "hash": "a019afdb34ee098237c58f23ab5e80e7a51c0e91a2abc8e809e234fa3782f121",
                "title": "Test News Item 11",
                "content": "San Francisco based cloud infrastructure giant experiences significant data breach exposing millions of user records. The associated threat actors appear to be nation-state sponsored.",
                "collected": "2023-08-02T14:05:04.801886",
                "published": "2023-08-02T14:05:04.801870",
                "updated": "2023-08-02T14:04:39.893435",
            },
        }
    ],
    "tags": {
        "cloud": {"name": "cloud", "tag_type": "MISC", "sub_forms": []},
        "data breach": {"name": "data breach", "tag_type": "MISC", "sub_forms": []},
        "nation-state": {"name": "nation-state", "tag_type": "MISC", "sub_forms": []},
    },
}

news_item_aggregate_6 = {
    "id": 6,
    "news_items": [
        {
            "id": 12,
            "news_item_data_id": "d45b6cde-f456-7890-91bc-3ed456f78901",
            "news_item_aggregate_id": 6,
            "news_item_data": {
                "id": "d45b6cde-f456-7890-91bc-3ed456f78901",
                "hash": "b029bfbb34ef098238c59f34bc5f90f8b52c1e91a3efc9e809e456fb4790f132",
                "title": "Test News Item 12",
                "content": "Sophisticated malware targeting Linux servers has been identified, with potentially devastating consequences for enterprise operations. Cybersecurity firms are actively working on mitigation strategies.",
                "collected": "2023-08-03T19:07:04.801886",
                "published": "2023-08-03T19:07:04.801870",
                "updated": "2023-08-03T19:06:39.893435",
            },
        }
    ],
    "tags": {
        "malware": {"name": "malware", "tag_type": "MISC", "sub_forms": []},
        "Linux": {"name": "Linux servers", "tag_type": "MISC", "sub_forms": []},
        "Cybersecurity": {"name": "Cybersecurity", "tag_type": "MISC", "sub_forms": []},
    },
}

news_item_aggregate_7 = {
    "id": 7,
    "news_items": [
        {
            "id": 137,
            "news_item_data_id": "e7890cde-g567-1234-56de-4fgh5678i901",
            "news_item_aggregate_id": 7,
            "news_item_data": {
                "id": "e7890cde-g567-1234-56de-4fgh5678i901",
                "hash": "c139cfcd35gf098239c58g45ch6g91h9j53d2e91a4fgh9e809e567gh5891i104",
                "title": "Test News Item",
                "content": "In a surprising move, a popular European software conglomerate admits to a long-standing vulnerability in its core codebase, urging users to apply patches immediately.",
                "collected": "2023-08-04T21:09:04.801886",
                "published": "2023-08-04T21:09:04.801870",
                "updated": "2023-08-04T21:08:39.893435",
            },
        }
    ],
    "tags": {
        "European software": {"name": "European software", "tag_type": "MISC", "sub_forms": []},
        "vulnerability": {"name": "vulnerability", "tag_type": "MISC", "sub_forms": []},
        "patches": {"name": "patches", "tag_type": "MISC", "sub_forms": []},
    },
}


news_item_aggregate_8 = {
    "id": 8,
    "news_items": [
        {
            "id": 4242,
            "news_item_data_id": "809f93ef-f00e-423b-89f8-59b917a9e039",
            "news_item_aggregate_id": 8,
            "news_item_data": {
                "id": "809f93ef-f00e-423b-89f8-59b917a9e039",
                "hash": "599fafee5eeb098239c57c78bf5cea6ea52b0e92a1abc9e80964150a3773f135",
                "title": "Test News Item",
                "content": "Microsoft Azure launches in Europe!",
                "collected": "2023-08-01T17:01:04.801886",
                "published": "2023-08-01T17:01:04.801870",
                "updated": "2023-08-01T17:00:39.893435",
            },
        }
    ],
    "tags": {
        "Azure": {"name": "Azure", "tag_type": "MISC", "sub_forms": []},
        "Europe": {"name": "Europe", "tag_type": "LOC", "sub_forms": []},
        "Microsoft": {"name": "Microsoft", "tag_type": "ORG", "sub_forms": []},
    },
}

news_item_aggregate_9 = {
    "id": 9,
    "news_items": [
        {
            "id": 23,
            "news_item_data_id": "809f93ef-f00e-423b-89f8-59b917a9e039",
            "news_item_aggregate_id": 9,
            "news_item_data": {
                "id": "809f93ef-f00e-423b-89f8-59b917a9e039",
                "hash": "599fafee5eeb098239c57c78bf5cea6ea52b0e92a1abc9e80964150a3773f135",
                "title": "Test News Item 10",
                "content": "An unspecified Software and OS developer from Redmond Washingigton has so many security issues you won't belive the 5th vulnerability - See the list of CVEs only from the last hour below: CVE-....",
                "collected": "2023-08-01T17:01:04.801886",
                "published": "2023-08-01T17:01:04.801870",
                "updated": "2023-08-01T17:00:39.893435",
            },
        }
    ],
    "tags": {
        "Software": {"name": "Software", "tag_type": "MISC", "sub_forms": []},
        "CVE": {"name": "CVE", "tag_type": "MISC", "sub_forms": []},
        "vulnerability": {"name": "vulnerability", "tag_type": "MISC", "sub_forms": []},
    },
}


news_item_aggregate_10 = {
    "id": 10,
    "news_items": [
        {
            "id": 1337,
            "news_item_data_id": "809f93ef-f00e-423b-89f8-59b917a9e039",
            "news_item_aggregate_id": 10,
            "news_item_data": {
                "id": "809f93ef-f00e-423b-89f8-59b917a9e039",
                "hash": "599fafee5eeb098239c57c78bf5cea6ea52b0e92a1abc9e80964150a3773f135",
                "title": "Test News Item 1337",
                "content": "Follow our new cybersec blog under http://<redacted>.com/blog for news on cybersecurity and the latest vulnerabilities.",
                "collected": "2023-08-01T17:01:04.801886",
                "published": "2023-08-01T17:01:04.801870",
                "updated": "2023-08-01T17:00:39.893435",
            },
        }
    ],
    "tags": {
        "Cybersecurity": {"name": "Cybersecurity", "tag_type": "MISC", "sub_forms": []},
        "blog": {"name": "blog", "tag_type": "MISC", "sub_forms": []},
        "vulnerability": {"name": "vulnerability", "tag_type": "MISC", "sub_forms": []},
    },
}


def merge_dicts(base_dict: dict, update_dict: dict):
    """Merge two nested dictionaries. If a key's value is a list in update_dict, apply the content of base_dict to every element of the list."""

    merged_dict = base_dict.copy()

    for key, value in update_dict.items():
        if key in merged_dict:
            # If both values are dictionaries, merge them
            if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                merged_dict[key] = merge_dicts(merged_dict[key], value)
            # If the value in update_dict the key is 'news_items', merge the content of base_dict's news_item_data to every element of the list in update_dict
            elif key == "news_items":
                base_data = merged_dict[key][0]["news_item_data"] if key in merged_dict else {}
                for idx in range(len(value)):
                    value[idx]["news_item_data"] = merge_dicts(base_data.copy(), value[idx].get("news_item_data", {}))
                merged_dict[key] = value  # Use the entire news_items list from update_dict
            else:
                merged_dict[key] = value
        else:
            merged_dict[key] = value

    return merged_dict


def merge_multiple(base_dict: dict, update_dicts: list[dict]):
    merged_dicts = []

    for update_dict in update_dicts:
        merged = merge_dicts(copy.deepcopy(base_dict), copy.deepcopy(update_dict))
        merged_dicts.append(merged)

    return merged_dicts


def cluster_news_items(news_item_aggregates: list[dict]):
    base_aggregate = copy.deepcopy(news_item_aggregates[0])
    for news_item_aggregate in news_item_aggregates[1:]:
        base_aggregate["news_items"].append(news_item_aggregate["news_items"][0])
    return base_aggregate


news_item_list = merge_multiple(
    base_aggregate,
    [
        news_item_aggregate_1,
        news_item_aggregate_2,
        news_item_aggregate_3,
        news_item_aggregate_4,
        news_item_aggregate_5,
        news_item_aggregate_6,
        news_item_aggregate_7,
        news_item_aggregate_8,
        news_item_aggregate_9,
        news_item_aggregate_10,
    ],
)

clustered_news_item_list = merge_multiple(
    base_aggregate,
    [
        cluster_news_items([news_item_aggregate_4, news_item_aggregate_7]),
        cluster_news_items([news_item_aggregate_7, news_item_aggregate_9, news_item_aggregate_10]),
        news_item_aggregate_1,
        news_item_aggregate_2,
        news_item_aggregate_3,
    ],
)

news_item_tags_1 = {"Cyber": {"name": "Cyber", "tag_type": "CySec", "sub_forms": ["CyberSecurity"]}}
news_item_tags_2 = {"Security": {"name": "Security", "tag_type": "Misc", "sub_forms": ["securities"]}}
news_item_tags_3 = {"New Orleans": {"name": "New Orleans", "tag_type": "LOC", "sub_forms": []}}
news_item_tags_4 = {"CVE": {"name": "CVE", "tag_type": "CySec", "sub_forms": ["cves"]}}
news_item_tags_5 = {"CVE-2021-1234": {"name": "CVE-2021-1234", "tag_type": "CVE", "sub_forms": []}}


if __name__ == "__main__":
    import json

    news_item_list_json = open("news_item_list.json", "w")
    json.dump(news_item_list, news_item_list_json, indent=2, sort_keys=False)
    clustered_news_item_list_json = open("clustered_news_item_list.json", "w")
    json.dump(
        clustered_news_item_list,
        clustered_news_item_list_json,
        indent=2,
        sort_keys=False,
    )
