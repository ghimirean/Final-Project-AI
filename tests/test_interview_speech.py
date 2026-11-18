"""Tests for Agent 4, step 2: speech (agents/interview_speech.py).
Offline: audio is a synthetic WAV made here, Whisper and the microphone are
fakes (the real ones are blocked in tests/conftest.py). All data is fake."""
import wave

import numpy as np
import pytest

import interview_speech as sp
from interview_speech import SpeechError, analyze_words, find_fillers, find_long_pauses


def w(text, start, end):
    return {"word": text, "start": start, "end": end}


def sentence(text, start=0.0, step=0.4):
    """Evenly spaced words: 'a b c' -> [{'word': 'a', 0.0-0.35}, ...]"""
    return [w(t, start + i * step, start + i * step + 0.35) for i, t in enumerate(text.split())]


# ---------- audio files ----------

def tone(seconds, rate=sp.SAMPLE_RATE, freq=440):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_write_and_load_wav_roundtrip(tmp_path):
    path = sp.write_wav(str(tmp_path / "a.wav"), tone(1.0))
    samples = sp.load_audio(path)
    assert samples.dtype == np.float32 and len(samples) == sp.SAMPLE_RATE
    assert abs(float(samples.max()) - 0.3) < 0.01


def test_load_audio_resamples_and_mixes_to_mono(tmp_path):
    path = str(tmp_path / "stereo.wav")
    left = (tone(1.0, rate=8000) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.column_stack([left, left]).tobytes())
    samples = sp.load_audio(path)
    assert len(samples) == sp.SAMPLE_RATE          # 1 s at 16 kHz after resampling, one channel


def test_missing_audio_file(tmp_path):
    with pytest.raises(SpeechError, match="not found"):
        sp.load_audio(str(tmp_path / "nope.wav"))


def test_non_wav_without_ffmpeg_gives_a_clear_error(tmp_path):
    path = tmp_path / "x.mp3"
    path.write_bytes(b"not audio")
    with pytest.raises(SpeechError, match="Could not read"):
        sp.load_audio(str(path))


# ---------- the microphone ----------

class FakeStream:
    def __init__(self, callback, **kw):
        self.callback, self.kw = callback, kw

    def __enter__(self):
        for _ in range(3):
            self.callback((tone(0.1) * 32767).astype(np.int16).reshape(-1, 1), 1600, None, None)
        return self

    def __exit__(self, *a):
        return False


class FakeSoundDevice:
    def InputStream(self, callback, **kw):
        assert kw["samplerate"] == sp.SAMPLE_RATE and kw["channels"] == 1
        return FakeStream(callback, **kw)


def test_record_audio_writes_a_wav_and_stops_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_sounddevice", lambda: FakeSoundDevice())
    path = str(tmp_path / "rec" / "q1.wav")
    seconds = sp.record_audio(path, stop=lambda: True)
    assert abs(seconds - 0.3) < 0.01
    assert len(sp.load_audio(path)) == int(0.3 * sp.SAMPLE_RATE)


def test_record_audio_with_no_microphone(tmp_path):
    with pytest.raises(SpeechError, match="microphone"):
        sp.record_audio(str(tmp_path / "q.wav"), stop=lambda: True)


def test_record_audio_that_captured_nothing(tmp_path, monkeypatch):
    class SilentStream(FakeStream):
        def __enter__(self):
            return self                      # the callback is never called

    class Empty(FakeSoundDevice):
        def InputStream(self, callback, **kw):
            return SilentStream(callback, **kw)
    monkeypatch.setattr(sp, "_sounddevice", lambda: Empty())
    with pytest.raises(SpeechError, match="recorded nothing"):
        sp.record_audio(str(tmp_path / "q.wav"), stop=lambda: True)


# ---------- Whisper (fake model) ----------

class FakeModel:
    def __init__(self, result):
        self.result, self.calls = result, []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        return self.result


WHISPER_RESULT = {"segments": [{"words": [{"word": " Hello", "start": 0.5, "end": 0.9},
                                          {"word": " there,", "start": 1.0, "end": 1.4},
                                          {"word": "  ", "start": 1.4, "end": 1.4}]},
                               {"words": [{"word": " Um", "start": 2.0, "end": 2.3}]}]}


def test_transcribe_returns_timed_words_and_asks_whisper_to_keep_fillers(monkeypatch):
    model = FakeModel(WHISPER_RESULT)
    monkeypatch.setattr(sp, "_load_whisper_model", lambda: model)
    words = sp.transcribe(tone(3.0))
    assert [x["word"] for x in words] == ["Hello", "there,", "Um"]      # blank word dropped, spaces stripped
    assert words[0] == {"word": "Hello", "start": 0.5, "end": 0.9}
    assert model.calls[0]["word_timestamps"] is True
    assert "Umm" in model.calls[0]["initial_prompt"]                  # the prompt that keeps "um" and "uh"


def test_very_short_audio_is_not_sent_to_whisper(monkeypatch):
    monkeypatch.setattr(sp, "_load_whisper_model", lambda: pytest.fail("Whisper must not run"))
    assert sp.transcribe(tone(0.1)) == []


def test_whisper_crash_becomes_a_speech_error(monkeypatch):
    class Broken:
        def transcribe(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(sp, "_load_whisper_model", lambda: Broken())
    with pytest.raises(SpeechError, match="Whisper failed"):
        sp.transcribe(tone(1.0))


def test_model_that_cannot_load_is_a_speech_error():
    with pytest.raises(SpeechError, match="Whisper"):
        sp.transcribe(tone(1.0))          # conftest blocks the real loader


# ---------- fillers ----------

def kinds(words):
    return [f["word"] for f in find_fillers(words)]


def test_hesitations_always_count():
    words = sentence("so Um I think uh we could, er, try hmm it")
    assert kinds(words) == ["um", "uh", "er", "hmm"]


def test_like_is_a_filler_only_when_set_off():
    assert kinds(sentence("I like Java and I would like to learn Docker")) == []
    assert kinds(sentence("It was, like, really hard")) == ["like"]
    # a pause right before it also sets it off
    assert kinds([w("It", 0, .2), w("was", .3, .5), w("like", 1.2, 1.4), w("hard", 1.5, 1.8)]) == ["like"]


def test_two_word_fillers():
    assert kinds(sentence("We shipped it, you know, on time")) == ["you know"]
    assert kinds(sentence("Do you know Java well")) == []
    assert kinds(sentence("It was sort of fine")) == []
    assert kinds(sentence("It was, sort of, fine")) == ["sort of"]


def test_so_at_the_start_of_a_sentence_after_a_period():
    assert kinds(sentence("We did it. So we moved on")) == ["so"]
    assert kinds(sentence("It was so good")) == []


def test_filler_time_is_the_word_start():
    assert find_fillers([w("Hi", 0, .3), w("um", 1.5, 1.8)]) == [{"word": "um", "time": 1.5}]


# ---------- pauses ----------

def test_long_pauses_are_gaps_between_words():
    words = [w("a", 0, .5), w("b", .6, 1.0), w("c", 3.5, 4.0), w("d", 4.1, 4.5), w("e", 7.0, 7.5)]
    pauses = find_long_pauses(words)
    assert pauses == [{"start": 1.0, "end": 3.5, "seconds": 2.5}, {"start": 4.5, "end": 7.0, "seconds": 2.5}]


def test_exactly_the_threshold_counts_and_less_does_not():
    assert len(find_long_pauses([w("a", 0, 1), w("b", 3.0, 4)])) == 1       # gap 2.0
    assert find_long_pauses([w("a", 0, 1), w("b", 2.9, 4)]) == []           # gap 1.9


def test_silence_before_the_first_word_is_not_a_pause():
    result = analyze_words([w("hello", 5.0, 5.4), w("there", 5.5, 5.9)], 8.0)
    assert result["long_pause_count"] == 0 and result["seconds_before_first_word"] == 5.0


# ---------- the whole analysis ----------

def test_analyze_words_numbers():
    words = sentence("I built a small tool, um, and it worked well", step=0.5)    # 10 words, 0.0 -> 4.85 s
    r = analyze_words(words, 6.0)
    assert r["word_count"] == 10 and r["speaking_seconds"] == 4.85
    assert r["words_per_minute"] == pytest.approx(10 / (4.85 / 60), abs=0.1)
    assert r["filler_count"] == 1 and r["fillers"] == {"um": 1}
    assert r["filler_rate_per_100_words"] == 10.0
    assert r["duration_seconds"] == 6.0
    assert r["transcript"].startswith("I built a small tool,")
    assert r["long_pause_count"] == 0


def test_too_few_words_gives_no_pace():
    r = analyze_words(sentence("Yes of course"), 3.0)
    assert r["word_count"] == 3 and r["words_per_minute"] is None


def test_silence_gives_zero_words_not_an_error():
    r = analyze_words([], 12.0)
    assert r["word_count"] == 0 and r["words_per_minute"] is None
    assert r["filler_rate_per_100_words"] == 0.0 and r["seconds_before_first_word"] is None
    assert r["transcript"] == "" and r["long_pauses"] == []


def test_analyze_audio_end_to_end_with_a_fake_model(tmp_path, monkeypatch):
    words = sentence("I worked on a project, like, with Java for two years", step=0.45)
    monkeypatch.setattr(sp, "transcribe", lambda samples: words)
    path = sp.write_wav(str(tmp_path / "answer.wav"), tone(8.0))
    r = sp.analyze_audio(path)
    assert r["duration_seconds"] == 8.0 and r["word_count"] == 11
    assert r["fillers"] == {"like": 1}
    assert 100 < r["words_per_minute"] < 160


def test_results_are_json_serialisable():
    import json
    json.dumps(analyze_words(sentence("a um b c d e f"), 5.0))
