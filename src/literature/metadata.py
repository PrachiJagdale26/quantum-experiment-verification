"""
Metadata normalisation and validation for scholarly records.
Deterministic: DOIs, arXiv ids and titles are normalised by rules, never guessed.
"""

import html
import re
import unicodedata
from typing import Optional, Tuple

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", re.I)


def normalize_doi(value: Optional[str]) -> Optional[str]:
    """'https://doi.org/10.1038/ABC' -> '10.1038/abc'. Returns None unless it is a syntactically valid DOI."""
    if not value:
        return None
    v = value.strip()
    v = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", v, flags=re.I)
    v = re.sub(r"^doi:\s*", "", v, flags=re.I)
    v = v.strip().rstrip(".").lower()
    return v if _DOI_RE.match(v) else None


def split_arxiv_id(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """'http://arxiv.org/abs/2110.05389v3' -> ('2110.05389', 'v3'). Old-style ids are supported."""
    if not value:
        return None, None
    v = value.strip()
    v = re.sub(r"^(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/", "", v, flags=re.I)
    v = re.sub(r"^arxiv:", "", v, flags=re.I).removesuffix(".pdf")
    for pat in (_ARXIV_NEW, _ARXIV_OLD):
        m = pat.search(v)
        if m:
            return m.group(1), m.group(2)
    return None, None


def normalize_title(title: Optional[str]) -> str:
    """Lower-case, accents and LaTeX/HTML markup removed, punctuation collapsed: used for matching only."""
    if not title:
        return ""
    t = html.unescape(title)
    t = re.sub(r"<[^>]+>", " ", t)                       # JATS / MathML tags
    t = re.sub(r"\$[^$]*\$", " ", t)                      # inline LaTeX
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t)       # \emph{x} -> x
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()[:500]


def clean_abstract(text: Optional[str]) -> Optional[str]:
    """Strip JATS/HTML markup that Crossref and others embed in abstracts. Empty -> None."""
    if not text:
        return None
    t = html.unescape(text)
    t = re.sub(r"<jats:title>[^<]*</jats:title>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def reconstruct_inverted_abstract(index: Optional[dict]) -> Optional[str]:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the original word order."""
    if not index:
        return None
    positions = [(pos, word) for word, poss in index.items() for pos in poss]
    if not positions:
        return None
    return " ".join(word for _, word in sorted(positions))


def first_author_surname(authors) -> Optional[str]:
    if not authors:
        return None
    name = str(authors[0]).strip()
    if "," in name:
        return normalize_title(name.split(",")[0]) or None
    parts = name.split()
    return normalize_title(parts[-1]) if parts else None


def parse_year(value) -> Optional[int]:
    if value is None:
        return None
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None
