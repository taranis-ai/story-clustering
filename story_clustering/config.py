from typing import Literal
from taranis_base_bot.config import CommonSettings


class Settings(CommonSettings):
    MODEL: Literal["louvain"] = "louvain"
    PACKAGE_NAME: str = "story_clustering"
    HF_MODEL_INFO: bool = True
    PAYLOAD_SCHEMA: dict[str, dict] = {"stories": {"type": "list", "required": True}}


Config = Settings()
