"""Voice commands: push-to-talk -> local ASR -> intent.

The parser is pure and tested. ASR is faster-whisper on the local GPU (no cloud),
imported lazily so the rest of the app runs without it.

Supported (EN + PT-BR, because that is how I actually talk to it):
    "find a red forklift" / "procura uma empilhadeira"   -> search
    "spawn" / "spawn three" / "coloca" / "spawn number two" -> spawn
    "next" / "previous" / "próxima" / "anterior"         -> page
    "clear" / "limpa"                                    -> clear spawned
    "reset" / "recentra"                                 -> recenter user
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3, "quatro": 4, "cinco": 5,
        "seis": 6, "sete": 7, "oito": 8}

_SEARCH = re.compile(
    r"^(?:please\s+)?(?:find|search(?:\s+for)?|look\s+for|show(?:\s+me)?|procura(?:r)?|busca(?:r)?|mostra(?:r)?)\s+(?P<q>.+)$")
_SPAWN = re.compile(r"^(?:spawn|place|add|coloca(?:r)?|adiciona(?:r)?|cria(?:r)?)(?:\s+(?:number|número|numero|o|a))?(?:\s+(?P<n>\w+))?$")
_ARTICLES = re.compile(r"^(?:a|an|the|some|um|uma|o|a|os|as|uns|umas)\s+")


@dataclass(frozen=True)
class Intent:
    action: str            # search | spawn | next | previous | clear | recenter | unknown
    query: str | None = None
    index: int | None = None   # 1-based card number for spawn


def parse(text: str) -> Intent:
    t = re.sub(r"[.!?,]", "", text.strip().lower())
    t = re.sub(r"\s+", " ", t)
    if not t:
        return Intent("unknown")
    if t in ("next", "next page", "próxima", "proxima", "próxima página"):
        return Intent("next")
    if t in ("previous", "back", "anterior", "volta"):
        return Intent("previous")
    if t in ("clear", "clear all", "limpa", "limpar", "limpa tudo"):
        return Intent("clear")
    if t in ("reset", "recenter", "recentra", "recentrar"):
        return Intent("recenter")
    m = _SPAWN.match(t)
    if m:
        n = m["n"]
        idx = None
        if n:
            idx = int(n) if n.isdigit() else _NUM.get(n)
            if idx is None:
                return Intent("unknown")
        return Intent("spawn", index=idx)
    m = _SEARCH.match(t)
    if m:
        q = _ARTICLES.sub("", m["q"]).strip()
        return Intent("search", query=q) if q else Intent("unknown")
    return Intent("unknown")


class Transcriber:
    """faster-whisper wrapper. ``small`` is enough for short commands and loads in ~2 s on an RTX card."""

    def __init__(self, model: str = "small", device: str = "cuda"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")

    def transcribe(self, audio_f32_16k) -> str:
        segments, _ = self.model.transcribe(audio_f32_16k, beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()


class PushToTalk:
    """Records while ``start()``..``stop()`` (bound to a controller button)."""

    def __init__(self, samplerate: int = 16000):
        import sounddevice as sd
        self.sd, self.sr, self.buf, self.stream = sd, samplerate, [], None

    def start(self):
        import numpy as np
        self.buf = []
        self.stream = self.sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                          callback=lambda d, *_: self.buf.append(np.copy(d[:, 0])))
        self.stream.start()

    def stop(self):
        import numpy as np
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return np.concatenate(self.buf) if self.buf else np.zeros(0, dtype="float32")
