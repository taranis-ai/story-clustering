import copy

base_story = {
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
            "review": "",
            "author": "",
            "source": "https://url/",
            "link": "https://url/",
            "language": None,
            "osint_source_id": "78049551-dcef-45bd-a5cd-4fe842c4d5e2",
        }
    ],
    "tags": {},
}


story_1 = {
    "id": "1",
    "news_items": [
        {
            "id": "4b9a5a9e-04d7-41fc-928f-99e5ad608ebb",
            "story_id": "1",
            "hash": "a96e88baaff421165e90ac4bb9059971b86f88d5c2abba36d78a1264fb8e9c87",
            "title": "Test News Item 13",
            "content": "Microsoft announced a security update addressing CVE-2020-1234. Experts at Google found vulnerabilities impacting Linux systems. Cisco advises users to update their security protocols to prevent potential breaches. The security community is on alert for new threats.",
            "collected": "2023-08-01T17:01:04.802015",
            "published": "2023-08-01T17:01:04.801998",
            "updated": "2023-08-01T17:00:39.893435",
        }
    ],
    "tags": {
        "Microsoft": "Vendor",
        "security": "MISC",
        "CVE-2020-1234": "CVE",
        "Google": "Vendor",
        "vulnerabilities": "CySec",
        "Linux": "Product",
        "Cisco": "Vendor",
        "protocols": "MISC",
        "breaches": "CySec",
        "community": "MISC",
    },
}


story_2 = {
    "id": "2",
    "news_items": [
        {
            "id": "4b9a5a9e-04d7-41fc-928f-99e5ad608ebb",
            "story_id": "2",
            "hash": "a96e88baaff421165e90ac4bb9059971b86f88d5c2abba36d78a1264fb8e9c87",
            "title": "Test News Item 27",
            "content": "Intel collaborates with Oracle to mitigate CVE-2021-5678 vulnerabilities in cloud infrastructures. Meanwhile, Apple is focusing on enhancing security measures against cyber threats, urging customers to update systems. All Windows installations will be affected by this.",
            "collected": "2023-08-01T17:01:04.802015",
            "published": "2023-08-01T17:01:04.801998",
            "updated": "2023-08-01T17:00:39.893435",
        }
    ],
    "tags": {
        "Intel": "Vendor",
        "Oracle": "Vendor",
        "CVE-2021-5678": "CVE",
        "vulnerabilities": "MISC",
        "cloud": "MISC",
        "infrastructures": "MISC",
        "Apple": "Vendor",
        "security": "MISC",
        "Windows": "Product",
        "systems": "MISC",
    },
}

story_3 = {
    "id": "3",
    "news_items": [
        {
            "id": "533086da-c8c1-4f8e-b3ee-103268983580",
            "story_id": "3",
            "hash": "f4c7b52ecfe6ab612db30e7fa534b470fd11493fc92f30575577b356b2a1abc7",
            "title": "Test News Item",
            "content": "IBM has partnered with NVIDIA to tackle the vulnerabilities found in AI modules. In addition, Amazon is improving its security posture to counteract cyber attacks on AWS platforms.",
            "collected": "2023-08-01T17:01:04.801951",
            "published": "2023-08-01T17:01:04.801934",
            "updated": "2023-08-01T17:00:39.893435",
        }
    ],
    "tags": {
        "IBM": "Vendor",
        "NVIDIA": "Vendor",
        "vulnerabilities": "MISC",
        "AI": "MISC",
        "modules": "MISC",
        "Amazon": "Vendor",
        "security": "MISC",
        "cyber": "MISC",
        "attacks": "MISC",
        "AWS": "Vendor",
    },
}

story_4 = {
    "id": "4",
    "news_items": [
        {
            "id": "f8912dab-h345-6789-01jk-5lmn6789o012",
            "story_id": "4",
            "hash": "d250dgfg45gh09824j59h56jk7k82l0m53o4p91q5rst9u809v567wx8910y234",
            "title": "Test News Item",
            "content": "Breaking: Local tech wizard claims to have found the solution to the world's biggest cybersecurity threat – turning it off and then back on again! Industries are flabbergasted by the simplicity of the remedy. Next he will fly to the moon with his rocket called 'Thunderbird 3'. His wife, Lady Penelope, a DevOps engineer, will accompany him.",
            "collected": "2023-08-05T23:11:04.801886",
            "published": "2023-08-05T23:11:04.801870",
            "updated": "2023-08-05T23:10:39.893435",
        }
    ],
    "tags": {
        "tech wizard": "MISC",
        "cybersecurity": "CySec",
        "remedy": "MISC",
        "Industries": "MISC",
        "solution": "MISC",
        "wizard": "MISC",
        "rocket": "MISC",
        "Thunderbird 3": "Product",
        "moon": "MISC",
        "Lady Penelope": "MISC",
        "DevOps": "MISC",
    },
}


story_5 = {
    "id": "5",
    "news_items": [
        {
            "id": "c12a3bde-a333-4567-90ab-2ed123f45678",
            "story_id": "5",
            "hash": "a019afdb34ee098237c58f23ab5e80e7a51c0e91a2abc8e809e234fa3782f121",
            "title": "Test News Item 11",
            "content": "Facebook initiated countermeasures against phishing attacks targeting user data. Following this, Salesforce has been investing in encryption technologies to protect customer information.",
            "collected": "2023-08-02T14:05:04.801886",
            "published": "2023-08-02T14:05:04.801870",
            "updated": "2023-08-02T14:04:39.893435",
        }
    ],
    "tags": {
        "Facebook": "Vendor",
        "countermeasures": "MISC",
        "phishing": "MISC",
        "attacks": "MISC",
        "user": "MISC",
        "data": "MISC",
        "Salesforce": "Vendor",
        "encryption": "MISC",
        "technologies": "MISC",
        "customer": "MISC",
    },
}

story_6 = {
    "id": "6",
    "news_items": [
        {
            "id": "d45b6cde-f456-7890-91bc-3ed456f78901",
            "story_id": "6",
            "hash": "b029bfbb34ef098238c59f34bc5f90f8b52c1e91a3efc9e809e456fb4790f132",
            "title": "Test News Item 12",
            "content": "Adobe is collaborating with SAP to address CVE-2023-7891 vulnerabilities in enterprise solutions. Additionally, Twitter is boosting its defense mechanisms against potential malware attacks targeting its infrastructure.",
            "collected": "2023-08-03T19:07:04.801886",
            "published": "2023-08-03T19:07:04.801870",
            "updated": "2023-08-03T19:06:39.893435",
        }
    ],
    "tags": {
        "Adobe": "MISC",
        "SAP": "MISC",
        "CVE-2023-7891": "CVE",
        "vulnerabilities": "MISC",
        "enterprise": "MISC",
        "solutions": "MISC",
        "Twitter": "MISC",
        "mechanisms": "MISC",
        "malware": "MISC",
        "infrastructure": "MISC",
    },
}

story_7 = {
    "id": "7",
    "news_items": [
        {
            "id": "e7890cde-g567-1234-56de-4fgh5678i901",
            "story_id": "7",
            "hash": "c139cfcd35gf098239c58g45ch6g91h9j53d2e91a4fgh9e809e567gh5891i104",
            "title": "Test News Item",
            "content": "HP is partnering with Qualcomm to develop firewall systems that counteract CVE-2023-1234 vulnerabilities in mobile devices. Concurrently, LinkedIn is working to fortify its security framework to shield against data breaches.",
            "collected": "2023-08-04T21:09:04.801886",
            "published": "2023-08-04T21:09:04.801870",
            "updated": "2023-08-04T21:08:39.893435",
        }
    ],
    "tags": {
        "HP": "Vendor",
        "Qualcomm": "MISC",
        "firewall": "MISC",
        "systems": "MISC",
        "CVE-2023-1234": "CVE",
        "vulnerabilities": "MISC",
        "mobile": "MISC",
        "devices": "MISC",
        "LinkedIn": "Vendor",
        "data": "MISC",
    },
}


story_8 = {
    "id": "8",
    "news_items": [
        {
            "id": "809f93ef-f00e-423b-89f8-59b917a9e039",
            "story_id": "8",
            "hash": "599fafee5eeb098239c57c78bf5cea6ea52b0e92a1abc9e80964150a3773f135",
            "title": "Test News Item",
            "content": "Dell has teamed up with VMware to combat security threats associated with CVE-2023-5678. At the same time, GitHub is heightening its security stance to fend off phishing scams targeting the platform.",
            "collected": "2023-08-01T17:01:04.801886",
            "published": "2023-08-01T17:01:04.801870",
            "updated": "2023-08-01T17:00:39.893435",
        }
    ],
    "tags": {
        "Dell": "Vendor",
        "VMware": "Vendor",
        "security": "MISC",
        "threats": "MISC",
        "CVE-2023-5678": "CVE",
        "GitHub": "Vendor",
        "stance": "MISC",
        "phishing": "MISC",
        "scams": "MISC",
        "platform": "MISC",
    },
}

story_9 = {
    "id": "9",
    "news_items": [
        {
            "id": "809f93ef-f00e-423b-89f8-59b917a9e039",
            "story_id": "9",
            "hash": "599fafee5eeb098239c57c78bf5cea6ea52b0e92a1abc9e80964150a3773f135",
            "title": "Test News Item 10",
            "content": "An unspecified Software and OS developer from Redmond Washingigton has so many security issues you won't belive the 5th vulnerability - See the list of CVEs only from the last hour below: CVE-....",
            "collected": "2023-08-01T17:01:04.801886",
            "published": "2023-08-01T17:01:04.801870",
            "updated": "2023-08-01T17:00:39.893435",
        }
    ],
    "tags": {
        "Software": "MISC",
        "CVE": "MISC",
        "vulnerability": "MISC",
    },
}


story_10 = {
    "id": "10",
    "news_items": [
        {
            "id": "809f93ef-f00e-423b-89f8-59b917a9e039",
            "story_id": "10",
            "hash": "599fafee5eeb098239c57c78bf5cea6ea52b0e92a1abc9e80964150a3773f135",
            "title": "Test News Item 1337",
            "content": "Follow our new cybersec blog under http://<redacted>.com/blog for news on cybersecurity and the latest vulnerabilities.",
            "collected": "2023-08-01T17:01:04.801886",
            "published": "2023-08-01T17:01:04.801870",
            "updated": "2023-08-01T17:00:39.893435",
        }
    ],
    "tags": {
        "Cybersecurity": "MISC",
        "blog": "MISC",
        "vulnerability": "MISC",
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
            # If the value in update_dict the key is 'news_items', merge the content of base_dict's to every element of the list
            elif key == "news_items":
                base_data = merged_dict[key][0] if key in merged_dict else {}
                for idx in range(len(value)):
                    value[idx] = merge_dicts(base_data.copy(), value[idx])
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


def cluster_news_items(storys: list[dict]):
    base_story = copy.deepcopy(storys[0])
    for story in storys[1:]:
        base_story["news_items"].append(story["news_items"][0])
    return base_story


news_item_list = merge_multiple(
    base_story,
    [
        story_1,
        story_2,
        story_3,
        story_4,
        story_5,
        story_6,
        story_7,
        story_8,
        story_9,
        story_10,
    ],
)

clustered_news_item_list = merge_multiple(
    base_story,
    [
        cluster_news_items([story_4, story_7]),
        cluster_news_items([story_7, story_9, story_10]),
        story_1,
        story_2,
        story_3,
    ],
)

news_item_tags_1 = {"Cyber": "CySec"}
news_item_tags_2 = {"Security": "Misc"}
news_item_tags_3 = {"New Orleans": "LOC"}
news_item_tags_4 = {"CVE": "CySec"}
news_item_tags_5 = {"CVE-2021-1234": "CVE"}


if __name__ == "__main__":
    import json

    news_item_list_json = open("story_list.json", "w")
    json.dump(news_item_list, news_item_list_json, indent=2, sort_keys=False)
    clustered_news_item_list_json = open("clustered_stories_list.json", "w")
    json.dump(
        clustered_news_item_list,
        clustered_news_item_list_json,
        indent=2,
        sort_keys=False,
    )
