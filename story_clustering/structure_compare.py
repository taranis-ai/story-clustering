import re
from difflib import SequenceMatcher


class StructureTokenizer:
    def __init__(self, short_threshold: int = 40):
        self.short_threshold = short_threshold

    def tokenize(self, text: str) -> list[str]:
        tokens = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip("\n")

            if not line.strip():
                tokens.append("BLANK")
                continue
            stripped = line.strip()

            if re.fullmatch(r"[-=*_]{3,}", stripped):
                tokens.append("RULER")
                continue

            if re.search(r"https?://\S+|\bwww\.\S+", stripped):
                tokens.append("URL")
                continue

            if raw_line.startswith(" " * 2) or raw_line.startswith("\t"):
                tokens.append("INDENT")
                continue

            if re.match(r"[*•\-+]\s+", stripped):
                tokens.append("UL")
                continue

            if re.match(r"\d+[\.\)]\s+", stripped):
                tokens.append("OL")
                continue

            if len(stripped) < self.short_threshold and (stripped.isupper() or stripped.endswith(":")):
                tokens.append("HEAD")
                continue

            if len(stripped) < self.short_threshold:
                tokens.append("SHORT")
                continue

            tokens.append("PARA")

        return tokens

    def calc_similarity(self, tokens_a: list[str], tokens_b: list[str]) -> float:
        return SequenceMatcher(None, tokens_a, tokens_b).ratio()

    def calc_fuzzy_similarity(self, tokens_a: list[str], tokens_b: list[str], window: int = 3) -> float:
        """
        A fuzzy structural comparison:
        Matching tokens within a small window count as partial matches.
        """

        score = 0
        total = max(len(tokens_a), len(tokens_b))

        for i, tok in enumerate(tokens_a):
            start = max(0, i - window)
            end = min(len(tokens_b), i + window + 1)
            if tok in tokens_b[start:end]:
                score += 1
        return score / total if total else 0
