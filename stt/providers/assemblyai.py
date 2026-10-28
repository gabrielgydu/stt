"""
AssemblyAI pre-recorded transcription: Universal-3.5 Pro with Universal-2 as
the fallback for languages 3.5 does not cover, speaker labels, and the key
terms as `keyterms_prompt`. Ported from audiorecorder, with the request fields
and key-term rules of claude-launcher's client.

Language: detection is always on. With language hints, they become the
expected languages and the first one is the fallback, which is also what turns
silence into a completed empty transcript instead of an error. Without hints,
detection runs over every language.

`discard` deletes the transcript and its audio. AssemblyAI refuses that while a
transcript is still processing, and an upload whose transcript was never
created cannot be deleted at all (AssemblyAI drops it on its own).
"""

from .. import config as _config
from .. import rest
from ..transcript import Transcript, Utterance
from . import Job, JobFailed, LeftBehind, Provider, audio_path, read_audio

DEFAULT_MODEL = "universal-3-5-pro"

# AssemblyAI's limits: 1000 key terms on universal-3-5-pro (universal-2 uses the
# first 200), at most six words per phrase. Its validation is loose (a
# seven-word phrase is accepted silently), so they are enforced here.
MAX_KEYTERMS = 1000
MAX_KEYTERM_WORDS = 6


def keyterms(terms):
    """The terms AssemblyAI is sent: no repeats, none over six words, the
    first 1000 (order = priority)."""
    terms = [t for t in _config.dedupe(terms or []) if len(t.split()) <= MAX_KEYTERM_WORDS]
    return terms[:MAX_KEYTERMS]


class AssemblyAI(Provider):
    name = "assemblyai"
    base_url = "https://api.assemblyai.com"
    max_seconds = 10 * 3600
    poll_seconds = 5

    def _headers(self):
        # The raw key: AssemblyAI takes no "Bearer" prefix.
        return {"Authorization": self.api_key}

    def submit(self, audio, *, speakers, terms, languages, progress):
        progress("Uploading audio to AssemblyAI...")
        uploaded = self.request(
            "POST", "/v2/upload", headers={"Content-Type": "application/octet-stream"},
            body=read_audio(audio), timeout=rest.UPLOAD_TIMEOUT,
        )
        if not isinstance(uploaded, dict) or not uploaded.get("upload_url"):
            raise ValueError("AssemblyAI upload: no upload_url in the answer")
        model = self.config.get("assemblyai_model") or DEFAULT_MODEL
        request = {
            "audio_url": uploaded["upload_url"],
            "speech_models": [model, "universal-2"] if model != "universal-2" else [model],
            "speaker_labels": bool(speakers),
            "language_detection": True,
        }
        if languages:
            request["language_detection_options"] = {
                "expected_languages": list(languages),
                "fallback_language": languages[0],
                "on_no_speech_detected": "fallback",
            }
        terms = keyterms(terms)
        if terms:
            request["keyterms_prompt"] = terms
        try:
            created = self.request("POST", "/v2/transcript", body=request)
        except rest.AnswerLost as e:
            # AssemblyAI may have created (and will bill) a transcript whose id
            # stt never saw: it cannot delete it, and trying again could
            # create a second one.
            raise LeftBehind(f"not retried: an AssemblyAI transcript may exist that stt cannot see; "
                             f"{e}") from None
        if not isinstance(created, dict) or not created.get("id"):
            raise ValueError("AssemblyAI submit: no transcript id in the answer")
        return Job(id=created["id"], path=audio_path(audio))

    def status(self, job):
        result = self.request("GET", f"/v2/transcript/{job.id}")
        if result["status"] == "error":
            raise JobFailed(self.scrub(result.get("error") or "transcription failed"))
        if result["status"] != "completed":
            return None
        return to_transcript(result)

    def discard(self, job):
        if not job.id:
            return None
        try:
            self.request("DELETE", f"/v2/transcript/{job.id}", timeout=rest.DISCARD_TIMEOUT)
        except rest.ApiError as e:
            if e.status == 404:
                return None
            # A transcript still processing answers 400 and runs on.
            return f"AssemblyAI kept the transcript (it refuses while one runs): {rest.clip(e, 120)}"
        except OSError as e:
            return f"could not delete the transcript: {rest.clip(e, 120)}"
        return None


def to_transcript(result):
    """Utterances from AssemblyAI's speaker labels; without them, one utterance
    for the whole text (as Soniox gives), so `utterances[0].language` works
    either way. Silence is an empty text and no utterances."""
    language = result.get("language_code")
    text = (result.get("text") or "").strip()
    utterances = [
        Utterance(
            text=u["text"],
            speaker=u.get("speaker"),
            start_ms=u.get("start"),
            end_ms=u.get("end"),
            language=language,
        )
        for u in result.get("utterances") or []
        if (u.get("text") or "").strip()
    ]
    if not utterances and text:
        words = result.get("words") or []
        utterances = [Utterance(
            text=text,
            start_ms=words[0].get("start") if words else None,
            end_ms=words[-1].get("end") if words else None,
            language=language,
        )]
    return Transcript(provider=AssemblyAI.name, text=text, utterances=utterances)
