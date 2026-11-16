"""Agent 3 key points with KeyBERT + embeddings (no LLM).

KeyBERT finds the phrases that best capture what a document is about:
it embeds the whole job description and every candidate phrase, then keeps
the phrases whose embeddings are closest to the document's embedding.
"Maximal Marginal Relevance" (MMR) makes the picks different from each
other, so we don't get five versions of "internship".

    1. Candidate phrases: the noun phrases in the posting ("OOP principles",
       "high traffic payment services"), found by spaCy's general English
       model (en_core_web_sm, already used by Agent 1). If spaCy is missing,
       KeyBERT's own 1-3 word groups are used instead.
    2. KeyBERT ranks them with Agent 3's embedding model (all-MiniLM-L6-v2,
       through embeddings.embed()), so it is the same local model that
       scores the jobs. Nothing leaves this computer.
    3. Each key phrase is compared with each resume line (embedding
       similarity) and the best line is kept. job_synthesizer decides what
       counts as covered: skills and degrees by exact rules, experience and
       role phrases by is_evidence() below, so a person can check the line.
"""
import re

from embeddings import EmbeddingError, embed

KEYPOINT_CONFIG = {
    "top_n": 6,           # key phrases per job
    "diversity": 0.5,     # MMR: 0 = most relevant only, 1 = most varied
    "spacy_model": "en_core_web_sm",
}

# EVIDENCE RULES for experience / role phrases (skills and degrees are
# decided by exact rules in job_synthesizer, never by similarity).
# Measured on the 3 fake fixture resumes x every non-skill phrase of the 6
# fixture jobs (102 pairs), after the degree/field phrases moved to rules:
#   true matches:  "dashboards" <- "Built weekly sales dashboards in Tableau" 0.54,
#                  "analytics" <- "Brightline Analytics | Junior Data Analyst" 0.53,
#                  "building websites" <- "Built client sites with React and Node.js." 0.47
#   false matches: "junior engineers" <- a degree line 0.44, "datasets" <- "Organized
#                  digital catalog records in Excel" 0.43, everything else lower
COVERED_AT = 0.45
# A multi-word phrase that shares exactly ONE word with the line is usually
# matched because of that word ("customer data" <- "and multi-customer
# support"), so it needs a clearly higher similarity.
SINGLE_SHARED_WORD_AT = 0.55

# Lines shorter than this are headings ("Work Experience") or fragments and
# match almost any job phrase, so they are never evidence.
MIN_LINE_WORDS = 4
BULLETS = "•▪◦●○■□➢►-–—*· \t"

# Words: keeps C++, C#, Node.js, .NET-style tokens together
TOKEN = r"[A-Za-z][A-Za-z0-9+#.]*[A-Za-z0-9+#]|[A-Za-z]"

# Words that say nothing about the job when they start or end a phrase
FILLER = {
    "ability", "bonus", "candidate", "candidates", "daily", "degree", "experience",
    "familiarity", "field", "hiring", "knowledge", "plus", "related", "requirement", "requirements",
    "role", "solid", "strong", "student", "students", "team", "understanding",
    "work", "year", "years",
}
MAX_PHRASE_WORDS = 5

_nlp = None
_keybert = None


def _stop_words():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return ENGLISH_STOP_WORDS


def _load_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(KEYPOINT_CONFIG["spacy_model"])
    return _nlp


def _load_keybert():
    # The only function that creates KeyBERT. Tests replace it (conftest.py).
    # KeyBERT gets our own embed() as its model, so it uses Agent 3's
    # sentence-transformers model and the same error handling.
    global _keybert
    if _keybert is None:
        try:
            from keybert import KeyBERT
            from keybert.backend import BaseEmbedder
        except Exception as e:
            raise EmbeddingError(f"KeyBERT not available: {type(e).__name__}") from None

        class OurModel(BaseEmbedder):
            def embed(self, documents, verbose=False):
                return embed(list(documents))

        _keybert = KeyBERT(model=OurModel())
    return _keybert


def candidate_phrases(text):
    """Noun phrases in the text, as {lowercase n-gram: phrase to show}.
    The key is the phrase the way KeyBERT's word counter sees it (stop words
    removed); the value is the readable original ("Design and train Deep
    Learning models"). Returns None if spaCy can't be used."""
    try:
        nlp = _load_spacy()
    except Exception:
        return None
    stop = _stop_words()
    phrases = {}
    for chunk in nlp(text).noun_chunks:
        tokens = [t for t in chunk if re.fullmatch(TOKEN, t.text)
                  and t.lower_ not in stop and t.pos_ not in ("DET", "PRON", "NUM")]
        while tokens and tokens[0].lower_ in FILLER:
            tokens = tokens[1:]
        while tokens and tokens[-1].lower_ in FILLER:
            tokens = tokens[:-1]
        if not tokens or len(tokens) > MAX_PHRASE_WORDS:
            continue
        key = " ".join(t.lower_ for t in tokens)
        shown = text[tokens[0].idx:tokens[-1].idx + len(tokens[-1].text)]
        phrases.setdefault(key, shown)
    return phrases


def extract_keyphrases(text, top_n=None):
    """The job's key phrases, best first: [(phrase, relevance 0-1), ...].
    Raises EmbeddingError if the embedding model can't be used."""
    from sklearn.feature_extraction.text import CountVectorizer
    top_n = top_n or KEYPOINT_CONFIG["top_n"]
    keybert = _load_keybert()
    stop = list(_stop_words())
    candidates = candidate_phrases(text)
    if candidates:
        longest = max(len(k.split()) for k in candidates)
        vectorizer = CountVectorizer(vocabulary=sorted(candidates), ngram_range=(1, longest),
                                     stop_words=stop, token_pattern=TOKEN)
    else:                                       # no spaCy: plain 1-3 word groups
        candidates = {}
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=stop + sorted(FILLER),
                                     token_pattern=TOKEN)
    try:
        pairs = keybert.extract_keywords(text, vectorizer=vectorizer, use_mmr=True,
                                         diversity=KEYPOINT_CONFIG["diversity"], top_n=top_n)
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"KeyBERT failed: {type(e).__name__}") from None
    return [(candidates.get(p, p), round(float(s), 3)) for p, s in pairs]


def resume_lines(text):
    """Resume lines usable as evidence. A line that starts in lowercase
    ("and multi-customer support") is the wrapped end of the line above, so
    it is joined back on. Bullet marks are removed; short lines are dropped."""
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if lines and line[0].islower():
            lines[-1] += " " + line
        else:
            lines.append(line.lstrip(BULLETS).strip())
    return [line for line in lines if len(line.split()) >= MIN_LINE_WORDS]


def content_words(text):
    return set(re.findall(r"[a-z0-9+#]+", text.lower())) - _stop_words()


def is_evidence(phrase, line, similarity):
    """Is this resume line real evidence for an experience / role phrase?"""
    if similarity < COVERED_AT:
        return False
    words = content_words(phrase)
    if len(words) >= 2 and len(words & content_words(line)) == 1:
        return similarity >= SINGLE_SHARED_WORD_AT
    return True


def match_phrases(phrases, lines, line_vectors=None):
    """For each key phrase, the most similar resume line.
    Returns [{"phrase", "relevance", "resume_similarity", "best_line"}].

    phrases: [(phrase, relevance)] from extract_keyphrases.
    line_vectors: embed(lines), passed in so a resume is embedded only once.
    """
    if not phrases:
        return []
    if lines and line_vectors is None:
        line_vectors = embed(lines)
    phrase_vectors = embed([p for p, _ in phrases])
    matches = []
    for i, (phrase, relevance) in enumerate(phrases):
        similarity, best_line = 0.0, None
        if lines:
            row = phrase_vectors[i] @ line_vectors.T     # unit vectors: dot = cosine
            best = int(row.argmax())
            similarity, best_line = float(row[best]), lines[best]
        matches.append({"phrase": phrase, "relevance": relevance,
                        "resume_similarity": round(similarity, 3), "best_line": best_line})
    return matches
