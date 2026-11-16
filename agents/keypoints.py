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
       similarity). A phrase counts as "covered" if the resume lists one of
       its skills or has a line at least COVERED_AT similar to it; the best
       line is kept as evidence so a person can check it.
"""
import re

from embeddings import EmbeddingError, embed
from resume_parser import extract_skills

KEYPOINT_CONFIG = {
    "top_n": 6,           # key phrases per job
    "diversity": 0.5,     # MMR: 0 = most relevant only, 1 = most varied
    "spacy_model": "en_core_web_sm",
}

# Phrase-to-resume-line similarity at or above this counts as covered.
# Measured on the 3 fake fixture resumes x 6 fixture jobs: real matches
# ("dashboards" for the data analyst 0.54, "building websites" for the web
# developer 0.47) were >= 0.45, while unrelated phrases were mostly below
# 0.40. A few related-but-different phrases land near it ("data science"
# vs. a "Computer Science" line: 0.53), so the matching line is always
# reported as evidence.
COVERED_AT = 0.45

# Resume lines shorter than this are headings ("Work Experience") and match
# almost any job phrase, so they are not used as evidence.
MIN_LINE_WORDS = 3

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
    return [line.strip() for line in (text or "").splitlines()
            if len(line.split()) >= MIN_LINE_WORDS]


def match_phrases(phrases, lines, candidate_skills, line_vectors=None):
    """For each key phrase: is it on the resume, and where?
    Returns [{"phrase", "relevance", "resume_similarity", "covered", "evidence"}].

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
        similarity, evidence = 0.0, None
        if lines:
            row = phrase_vectors[i] @ line_vectors.T     # unit vectors: dot = cosine
            best = int(row.argmax())
            similarity, evidence = float(row[best]), lines[best]
        skills = extract_skills(phrase, candidate_skills)
        if skills:                                     # an exact skill beats similarity
            covered, evidence = True, "skills: " + ", ".join(skills)
        else:
            covered = similarity >= COVERED_AT
        matches.append({"phrase": phrase, "relevance": relevance,
                        "resume_similarity": round(similarity, 3),
                        "covered": covered, "evidence": evidence if covered else None})
    return matches
