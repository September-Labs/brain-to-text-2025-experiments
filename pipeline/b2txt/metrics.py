import re
from typing import Iterable, Tuple

from b2txt.ctc import edit_distance


def normalize_sentence(sentence: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z\- '\"]", "", sentence)
    cleaned = cleaned.replace("- ", " ")
    cleaned = cleaned.replace("--", "")
    cleaned = cleaned.replace(" '", "'")
    cleaned = cleaned.lower().strip()
    cleaned = " ".join(word for word in cleaned.split() if word)
    return cleaned


def word_error_stats(true_sentence: str, pred_sentence: str) -> Tuple[int, int]:
    true_norm = normalize_sentence(true_sentence)
    pred_norm = normalize_sentence(pred_sentence)
    true_words = true_norm.split() if true_norm else []
    pred_words = pred_norm.split() if pred_norm else []
    edits = edit_distance(true_words, pred_words)
    return edits, len(true_words)
