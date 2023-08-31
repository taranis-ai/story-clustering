import nltk
import logging
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")

bert_logger = logging.getLogger("BERTopic")

stopwords_list = stopwords.words("english")
stopwords_list.extend(stopwords.words("german"))


class IgnoreSpecificLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Ran model with model id" not in record.getMessage()


bert_logger.addFilter(IgnoreSpecificLogFilter())

sentence_transformer = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

logger = logging.getLogger("story_clustering")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

logger.info("Story Clustering Setup Complete.")
