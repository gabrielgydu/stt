"""The provider-neutral result of a file transcription."""

from dataclasses import asdict, dataclass, field


@dataclass
class Utterance:
    text: str
    speaker: str | None = None   # "A", "B", ... in order of first appearance
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None  # ISO code, when the provider reports one


@dataclass
class Transcript:
    provider: str
    text: str
    utterances: list[Utterance] = field(default_factory=list)

    def format(self):
        """ "Speaker A: ..." blocks separated by blank lines when the provider
        separated speakers, else the plain text."""
        if any(u.speaker for u in self.utterances):
            return "\n\n".join(f"Speaker {u.speaker}: {u.text}" for u in self.utterances)
        return self.text

    def to_dict(self):
        return asdict(self)


def speaker_letter(index):
    """0 -> "A", 25 -> "Z", 26 -> "27": letters while they last, like AssemblyAI."""
    return chr(ord("A") + index) if index < 26 else str(index + 1)
