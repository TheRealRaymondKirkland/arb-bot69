"""
Improved title matcher for cross-platform market pairing.

Handles the key challenge: Kalshi titles like "Will the Fed cut rates in June 2026?"
vs Polymarket titles like "Federal Reserve June 2026 rate decision".
"""
import re
from typing import Tuple

STOP = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "will",
    "be", "by", "at", "on", "who", "what", "which", "next", "new",
    "there", "have", "this", "that", "with", "from", "are", "was", "were",
    "would", "could", "should", "its", "do", "does", "did", "has", "had",
    "any", "all", "not", "but", "if", "than", "then", "when", "where",
    "how", "about", "into", "during", "before", "after", "above", "below",
    "both", "each", "more", "most", "other", "some", "such", "no", "nor",
    "so", "yet", "either", "whether", "make", "just", "over", "get", "out",
    "use", "two", "first", "also", "back", "per", "vs", "via", "least",
    "whether", "least", "between", "among", "within", "without",
})

# Expand abbreviations into canonical tokens before matching
EXPANSIONS: dict[str, list[str]] = {
    "fed":      ["federal", "reserve"],
    "fomc":     ["federal", "reserve"],
    "bps":      ["basis", "point"],
    "bp":       ["basis", "point"],
    "gov":      ["government"],
    "govt":     ["government"],
    "pres":     ["president"],
    "sec":      ["secretary"],        # Press Secretary, Labor Secretary, etc.
    "secy":     ["secretary"],
    "dem":      ["democrat"],
    "rep":      ["republican"],
    "gop":      ["republican"],
    "uk":       ["united", "kingdom"],
    "eu":       ["european", "union"],
    "scotus":   ["supreme", "court"],
    "potus":    ["president", "united", "states"],
    "ag":       ["attorney", "general"],
    "doj":      ["department", "justice"],
    "gdp":      ["gdp"],
    "cpi":      ["cpi"],
    "nato":     ["nato"],
    "irs":      ["internal", "revenue"],
    "fbi":      ["federal", "bureau", "investigation"],
    "cia":      ["central", "intelligence"],
    "nasa":     ["nasa"],
    "fda":      ["food", "drug", "administration"],
}

MONTH_NORMS: dict[str, str] = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
    "jan": "jan", "feb": "feb", "mar": "mar", "apr": "apr",
    "jun": "jun", "jul": "jul", "aug": "aug", "sep": "sep",
    "oct": "oct", "nov": "nov", "dec": "dec",
}


def _stem(word: str) -> str:
    """Minimal suffix stripping so rate/rates, cut/cuts, hike/hikes match."""
    if len(word) <= 3:
        return word
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    if word.endswith("tion") and len(word) > 6:
        return word[:-4]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _normalize(tokens: list[str]) -> tuple[set[str], set[str]]:
    """Returns (content_tokens, temporal_tokens) after expansion + stemming."""
    expanded: list[str] = []
    for tok in tokens:
        if tok in EXPANSIONS:
            expanded.extend(EXPANSIONS[tok])
        else:
            expanded.append(tok)

    content: set[str] = set()
    temporal: set[str] = set()

    i = 0
    while i < len(expanded):
        tok = expanded[i]

        # Year token — may combine with adjacent month
        if re.match(r"^20\d{2}$", tok):
            nxt = expanded[i + 1] if i + 1 < len(expanded) else ""
            if nxt in MONTH_NORMS:
                temporal.add(f"{tok}_{MONTH_NORMS[nxt]}")
                i += 2
                continue
            temporal.add(tok)
            i += 1
            continue

        # Month token — may combine with adjacent year
        if tok in MONTH_NORMS:
            month_norm = MONTH_NORMS[tok]
            nxt = expanded[i + 1] if i + 1 < len(expanded) else ""
            if re.match(r"^20\d{2}$", nxt):
                temporal.add(f"{nxt}_{month_norm}")
                i += 2
                continue
            temporal.add(month_norm)
            i += 1
            continue

        if tok in STOP or len(tok) <= 1:
            i += 1
            continue

        content.add(_stem(tok))
        i += 1

    return content, temporal


def match_score(title_a: str, title_b: str) -> Tuple[float, float, float]:
    """
    Returns (combined_score, content_score, temporal_score).
    combined_score is 0-1; >= 0.30 is a candidate match.
    """
    content_a, temporal_a = _normalize(_tokenize(title_a))
    content_b, temporal_b = _normalize(_tokenize(title_b))

    # Content Jaccard
    union_c = content_a | content_b
    content_score = len(content_a & content_b) / len(union_c) if union_c else 0.0

    # Temporal score
    if temporal_a and temporal_b:
        union_t = temporal_a | temporal_b
        temporal_score = len(temporal_a & temporal_b) / len(union_t)
    elif not temporal_a and not temporal_b:
        temporal_score = 0.0  # no date info either way — don't boost
    else:
        temporal_score = 0.5  # one has dates, other doesn't — uncertain

    # When neither title has dates, rely entirely on content similarity
    if not temporal_a and not temporal_b:
        combined = content_score
    else:
        combined = 0.65 * content_score + 0.35 * temporal_score

    # Hard penalty: both sides have temporal tokens but zero overlap
    if temporal_a and temporal_b and not (temporal_a & temporal_b):
        years_a = {t.split("_")[0] for t in temporal_a}
        years_b = {t.split("_")[0] for t in temporal_b}
        if years_a & years_b:
            combined *= 0.75  # same year, different month — mild penalty
        else:
            combined *= 0.30  # different year entirely — almost certainly wrong

    return combined, content_score, temporal_score
