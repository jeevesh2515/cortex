"""Query understanding.

BM25 treats every term as equally important, so a question phrased as a question
is largely stopwords: "what did I decide about the chunking strategy for my
notes" is thirteen tokens of which three carry signal. The filler does not just
waste effort, it actively dilutes the scoring.

The standard fix is an LLM call to extract salient terms and generate
alternative phrasings -- obsidian-copilot's approach. Cortex does it locally
instead, for three reasons: it must work offline; on a fanless machine an LLM
call before every search is a real latency and thermal cost; and the win here
comes mostly from *removing* filler rather than inventing synonyms, which needs
no model at all.

Three signals are extracted:

* **Salient terms** -- content words, weighted towards the rare and the
  distinctive. These become a second, tighter lexical query.
* **Quoted phrases** -- an explicit request for exact matching, which must not
  be diluted by expansion.
* **Tags** -- ``#project/cortex`` in a query is a filter the user typed by hand,
  and far more precise than anything retrieval would infer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["STOPWORDS", "ExpandedQuery", "expand_query"]

# fmt: off
_STOPWORD_TEXT = """
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn
do does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into
is isn it its itself just me more most my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan she should shouldn
so some such than that the their theirs them themselves then there these they
this those through to too under until up very was wasn we were weren what when
where which while who whom why will with won would wouldn you your yours yourself
yourselves tell show find get give need want know think say said please help
anything something note notes vault file files document documents
"""
# fmt: on

STOPWORDS = frozenset(_STOPWORD_TEXT.split())
"""Filler words carrying no signal in a personal notes system.

Deliberately includes interrogatives and first-person pronouns: in a general
search engine "what" and "my" carry a little signal, but here every note is
implicitly the user's, so they only serve to match everything.

Also includes the domain's own vocabulary -- "note", "notes", "vault",
"document". These are long enough to pass the distinctiveness heuristic and so
would otherwise be ranked as *leading* salient terms, while matching literally
every document in the corpus. Caught by test: "notes on LanceDB" ranked "notes"
above "LanceDB".
"""


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_QUOTED = re.compile(r'"([^"]{2,})"|“([^”]{2,})”')
_TAG = re.compile(r"(?:(?<=\s)|(?<=^))#([A-Za-z0-9_/-]*[A-Za-z_/-][A-Za-z0-9_/-]*)")


@dataclass(slots=True)
class ExpandedQuery:
    """A query broken into the signals retrieval can actually use."""

    original: str
    salient_terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def salient_query(self) -> str:
        """Content words only, as a second lexical query."""
        return " ".join(self.salient_terms)

    @property
    def has_signal(self) -> bool:
        """Whether expansion produced anything worth a separate search.

        A single salient term is already what BM25 would score on, so running a
        second identical query would just double the work for no gain.
        """
        return len(self.salient_terms) >= 2 or bool(self.phrases)

    def lexical_variants(self) -> list[str]:
        """Distinct lexical queries to run, strongest intent first.

        Quoted phrases come first: the user asked for them explicitly, and
        nothing inferred should outrank an instruction.
        """
        variants: list[str] = []
        for phrase in self.phrases:
            if phrase not in variants:
                variants.append(phrase)
        if self.has_signal:
            salient = self.salient_query
            if salient and salient not in variants:
                variants.append(salient)
        if self.original.strip() and self.original.strip() not in variants:
            variants.append(self.original.strip())
        return variants


def _is_distinctive(term: str) -> bool:
    """Whether a term looks like it carries meaning.

    Heuristics rather than a model: long words, anything containing a digit or
    an internal capital (``LanceDB``, ``qwen3``, ``k8s``), and hyphenated or
    underscored compounds are all more distinctive than short common words.
    """
    if len(term) >= 5:
        return True
    if any(char.isdigit() for char in term):
        return True
    if any(char.isupper() for char in term[1:]):
        return True
    return "-" in term or "_" in term


def expand_query(query: str, *, max_terms: int = 12) -> ExpandedQuery:
    """Break a query into salient terms, quoted phrases and tags."""
    original = query.strip()
    if not original:
        return ExpandedQuery(original="")

    phrases = [
        (first or second).strip()
        for first, second in _QUOTED.findall(original)
        if (first or second).strip()
    ]

    tags: list[str] = []
    for match in _TAG.finditer(original):
        tag = match.group(1).rstrip("/")
        if tag and tag not in tags:
            tags.append(tag)

    # Quoted spans and tags are already accounted for; drop them so they do not
    # also arrive as loose salient terms.
    remainder = _QUOTED.sub(" ", original)
    remainder = _TAG.sub(" ", remainder)

    seen: set[str] = set()
    ordered: list[tuple[bool, str]] = []
    for token in _WORD.findall(remainder):
        lowered = token.lower().strip("'-_")
        if not lowered or lowered in STOPWORDS or lowered in seen:
            continue
        if len(lowered) < 2:
            continue
        seen.add(lowered)
        ordered.append((_is_distinctive(token), lowered))

    # Distinctive terms first, original order preserved within each group, so the
    # tighter query leads with the words most likely to discriminate.
    salient = [term for distinctive, term in ordered if distinctive]
    salient += [term for distinctive, term in ordered if not distinctive]

    return ExpandedQuery(
        original=original,
        salient_terms=salient[:max_terms],
        phrases=phrases,
        tags=tags,
    )
