import pytest
import json


@pytest.fixture(scope="session")
def story_list():
    with open("./story_list.json", "r") as f:
        yield json.load(f)


@pytest.fixture(scope="session")
def clustered_stories_list():
    with open("./clustered_stories_list.json", "r") as f:
        yield json.load(f)
