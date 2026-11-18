"""Agent 4, step 2: speech. Records an answer (or takes a pre-recorded audio
file), transcribes it with Whisper on this computer, and measures how it was said.

    record_audio(path)            microphone -> 16 kHz mono WAV
    transcribe(audio_path)        Whisper -> words with start/end times
    analyze_words(words, secs)    PURE function: words per minute, filler words, long pauses
    analyze_audio(audio_path)     transcribe + analyze_words (what the interview calls)

Whisper and sounddevice are imported only when used, so importing this file
is cheap and the tests never load a model or open a microphone (tests/conftest.py).
Nothing is sent anywhere: Whisper runs locally (the model downloads once, on first use).
"""
import os
import re
import wave

import numpy as np

SPEECH_CONFIG = {"model": "base.en", "language": "en"}   # Whisper model; base.en = ~140 MB, fine on a laptop CPU
SAMPLE_RATE = 16000                # Whisper's rate
LONG_PAUSE_SECONDS = 2.0           # a silence between two words this long counts as a long pause
MIN_WORDS_FOR_PACE = 5             # fewer words than this: words per minute is not meaningful (None)
MAX_RECORD_SECONDS = 120

# Whisper tends to leave "um" and "uh" out of transcripts. A prompt that contains them makes it keep them.
WHISPER_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, here's what I think. You know, I mean, it's, like, fine."

HESITATIONS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "hm", "mm", "mmm", "eh"}
# words that are fillers only when set off from the sentence (commas or a pause around them):
# "I like Java" is not a filler, "It was, like, hard" is
DISCOURSE_FILLERS = {"like", "basically", "literally", "actually", "so"}
DISCOURSE_PHRASES = [("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")]
SET_OFF_PAUSE = 0.3                # seconds of silence before/after a word that also "set it off"

WORD_EDGE = re.compile(r"^\W+|\W+$")


class SpeechError(Exception):
    """Recording or transcription failed. The message says what to do."""


# ---------- audio in and out ----------

def _sounddevice():
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError) as e:        # OSError: PortAudio library missing
        raise SpeechError("Recording needs the 'sounddevice' package and a microphone "
                          f"(pip install sounddevice; on Linux also PortAudio): {e}")


def write_wav(path, samples, rate=SAMPLE_RATE):
    """Writes float samples (-1..1) or int16 samples as a 16-bit mono WAV."""
    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.tobytes())
    return path


def record_audio(path, max_seconds=MAX_RECORD_SECONDS, stop=None):
    """Records the microphone to a WAV file until `stop()` returns True (for
    example "the user pressed Enter") or max_seconds pass. Returns the
    duration in seconds. `stop` is polled about 10 times a second."""
    import time
    sd = _sounddevice()
    chunks = []
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            callback=lambda data, frames, t, status: chunks.append(data.copy())):
            started = time.time()
            while time.time() - started < max_seconds and not (stop and stop()):
                time.sleep(0.1)
    except Exception as e:                      # no microphone, permission denied, device busy
        raise SpeechError(f"Could not record from the microphone: {e}")
    if not chunks:
        raise SpeechError("The microphone recorded nothing. Check that it is connected and allowed.")
    samples = np.concatenate(chunks).reshape(-1)
    write_wav(path, samples)
    return len(samples) / SAMPLE_RATE


def load_audio(path):
    """Audio file -> float32 mono samples at 16 kHz. WAV is read here (no
    ffmpeg needed); other formats (mp3, m4a, ...) go through Whisper's loader, which needs ffmpeg."""
    if not os.path.isfile(path):
        raise SpeechError(f"Audio file not found: {path}")
    try:
        with wave.open(path, "rb") as f:
            rate, channels, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
            raw = f.readframes(f.getnframes())
    except wave.Error:
        return _load_with_ffmpeg(path)
    if width != 2:
        return _load_with_ffmpeg(path)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(samples):
        target = int(round(len(samples) * SAMPLE_RATE / rate))
        samples = np.interp(np.linspace(0, len(samples) - 1, target), np.arange(len(samples)), samples)
    return samples.astype(np.float32)


def _load_with_ffmpeg(path):
    try:
        import whisper
        return whisper.load_audio(path)
    except Exception as e:
        raise SpeechError(f"Could not read {path} (WAV is read directly; other formats need ffmpeg): {e}")


# ---------- Whisper ----------

_model_cache = {}


def _load_whisper_model():
    """Loads the Whisper model (cached for the run). The only place that imports whisper."""
    name = SPEECH_CONFIG["model"]
    if name not in _model_cache:
        try:
            import whisper
            _model_cache[name] = whisper.load_model(name)
        except Exception as e:
            raise SpeechError(f"Could not load the Whisper model '{name}' (it downloads on first use, so "
                              f"it needs internet once): {e}")
    return _model_cache[name]


def transcribe(samples):
    """Whisper on 16 kHz samples -> list of {"word", "start", "end"} (words as
    Whisper wrote them, with punctuation, so commas can show a filler)."""
    if len(samples) < SAMPLE_RATE * 0.3:
        return []
    model = _load_whisper_model()
    try:
        result = model.transcribe(samples, language=SPEECH_CONFIG["language"], word_timestamps=True,
                                  initial_prompt=WHISPER_PROMPT, condition_on_previous_text=False,
                                  fp16=False)
    except Exception as e:
        raise SpeechError(f"Whisper failed: {e}")
    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = str(w.get("word", "")).strip()
            if text:
                words.append({"word": text, "start": float(w["start"]), "end": float(w["end"])})
    return words


# ---------- the measurements (pure, no models) ----------

def normalize(word):
    return WORD_EDGE.sub("", str(word)).lower()


def find_fillers(words):
    """Returns [{"word", "time"}] for every filler word.
    - hesitations (um, uh, er, hmm, ...) always count
    - "like", "basically", "so", ... and "you know", "I mean", "kind of" count only
      when set off from the sentence: a comma or ellipsis next to them, or a
      pause of at least SET_OFF_PAUSE seconds on either side. "I like Java" is not a filler."""
    fillers, skip = [], set()
    count = len(words)

    def set_off(i, j):
        """Is the span words[i..j] set off by punctuation or a pause?"""
        before_punct = i > 0 and re.search(r"[,.;:…!?-]$", words[i - 1]["word"])
        after_punct = re.search(r"[,.;:…!?-]$", words[j]["word"])
        before_gap = i > 0 and words[i]["start"] - words[i - 1]["end"] >= SET_OFF_PAUSE
        after_gap = j + 1 < count and words[j + 1]["start"] - words[j]["end"] >= SET_OFF_PAUSE
        return bool(before_punct or after_punct or before_gap or after_gap)

    for i, w in enumerate(words):
        if i in skip:
            continue
        token = normalize(w["word"])
        if token in HESITATIONS:
            fillers.append({"word": token, "time": round(w["start"], 2)})
            continue
        for first, second in DISCOURSE_PHRASES:
            if token == first and i + 1 < count and normalize(words[i + 1]["word"]) == second:
                if set_off(i, i + 1):
                    fillers.append({"word": f"{first} {second}", "time": round(w["start"], 2)})
                    skip.add(i + 1)
                break
        else:
            if token in DISCOURSE_FILLERS and set_off(i, i):
                fillers.append({"word": token, "time": round(w["start"], 2)})
    return fillers


def find_long_pauses(words, threshold=LONG_PAUSE_SECONDS):
    """Silences of at least `threshold` seconds BETWEEN two words."""
    pauses = []
    for before, after in zip(words, words[1:]):
        gap = after["start"] - before["end"]
        if gap >= threshold:
            pauses.append({"start": round(before["end"], 2), "end": round(after["start"], 2),
                           "seconds": round(gap, 2)})
    return pauses


def analyze_words(words, duration_seconds):
    """Speech metrics for one answer from Whisper's word list.
    words_per_minute: words divided by the time from the first word's start to
    the last word's end (None for fewer than MIN_WORDS_FOR_PACE words).
    filler_rate: fillers per 100 words."""
    duration = round(float(duration_seconds or 0), 2)
    transcript = " ".join(w["word"] for w in words)
    word_count = len(words)
    fillers = find_fillers(words)
    pauses = find_long_pauses(words)
    pace = None
    speaking = 0.0
    if word_count:
        speaking = max(words[-1]["end"] - words[0]["start"], 0.0)
    if word_count >= MIN_WORDS_FOR_PACE and speaking > 0:
        pace = round(word_count / (speaking / 60.0), 1)
    by_word = {}
    for f in fillers:
        by_word[f["word"]] = by_word.get(f["word"], 0) + 1
    return {
        "transcript": transcript,
        "duration_seconds": duration,
        "word_count": word_count,
        "words_per_minute": pace,
        "filler_count": len(fillers),
        "filler_rate_per_100_words": round(100 * len(fillers) / word_count, 1) if word_count else 0.0,
        "fillers": by_word,
        "long_pause_count": len(pauses),
        "long_pauses": pauses,
        "seconds_before_first_word": round(words[0]["start"], 2) if word_count else None,
    }


def analyze_audio(audio_path):
    """Audio file -> speech metrics (see analyze_words). An answer with no
    speech gives zero words, not an error."""
    samples = load_audio(audio_path)
    duration = len(samples) / SAMPLE_RATE
    return analyze_words(transcribe(samples), duration)
