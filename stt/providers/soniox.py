"""
Soniox async transcription (https://soniox.com/docs/stt/async/async-transcription).

Flow: upload the file, create a transcription, poll it, fetch the tokens, then
delete both the transcription and the file. Soniox keeps uploads until they are
deleted (10 GB / 1,000 files / 2,000 transcriptions per account, purged after
30 days), so `discard` is not optional housekeeping.

Speaker labels, language identification and smart formatting are included in
the hourly price; the key terms go into `context.terms`.
"""

from .. import config as _config
from .. import rest
from ..transcript import Transcript, Utterance, speaker_letter
from . import Job, JobFailed, LeftBehind, Provider, audio_name, audio_path, read_audio

DEFAULT_MODEL = "stt-async-v5"


class Soniox(Provider):
    name = "soniox"
    base_url = "https://api.soniox.com"
    max_seconds = 300 * 60   # fixed per-file limit, not raisable
    poll_seconds = 3

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def submit(self, audio, *, speakers, terms, languages, progress):
        progress("Uploading audio to Soniox...")
        body, content_type = rest.multipart("file", audio_name(audio), read_audio(audio))
        uploaded = self.request(
            "POST", "/v1/files", headers={"Content-Type": content_type},
            body=body, timeout=rest.UPLOAD_TIMEOUT,
        )
        job = Job(id="", path=audio_path(audio), extra={"file_id": uploaded["id"]})

        request = {
            "model": self.config.get("soniox_model") or DEFAULT_MODEL,
            "file_id": uploaded["id"],
            "enable_speaker_diarization": bool(speakers),
            "enable_language_identification": True,
        }
        if languages:
            request["language_hints"] = languages
        terms = _config.dedupe(terms or [])
        if terms:
            request["context"] = {"terms": terms}
        try:
            created = self.request("POST", "/v1/transcriptions", body=request)
            job.id = created["id"]
        except BaseException as e:
            # The upload would otherwise linger for 30 days. BaseException, so
            # Ctrl-C or SIGTERM between the two requests cleans up too.
            left = self.cleanup(job)
            if isinstance(e, rest.AnswerLost):
                # Soniox may have created the transcription; stt never saw its
                # id, so it cannot delete it, and trying again could make a
                # second one. Without its file a pending one fails.
                gone = left or "its file was deleted, so it fails"
                # The marker goes first: the line is clipped at ~300
                # characters and a provider error can be long (an HTML page).
                raise LeftBehind(f"not retried: a Soniox transcription may exist that stt cannot see "
                                 f"({gone}); {e}") from None
            if left and rest.is_transient(e):
                raise LeftBehind(f"not retried, so no more uploads pile up: {left}; {e}") from None
            if left:
                progress(f"Could not delete the Soniox upload {job.extra['file_id']}: {left}")
            raise
        return job

    def status(self, job):
        result = self.request("GET", f"/v1/transcriptions/{job.id}")
        # "error" in the async guide, "failed" in the error reference: both final.
        if result["status"] in ("error", "failed"):
            raise JobFailed(self.scrub(result.get("error_message") or "transcription failed"))
        if result["status"] != "completed":
            return None
        transcript = self.request("GET", f"/v1/transcriptions/{job.id}/transcript")
        return to_transcript((transcript or {}).get("tokens") or [])

    def discard(self, job):
        # The transcription first: Soniox refuses (409) to delete one that is
        # still running, and says so. The file goes either way; a pending
        # transcription then fails for lack of audio, which is what we want.
        left = []
        if job.id:
            try:
                self.request("DELETE", f"/v1/transcriptions/{job.id}", timeout=rest.DISCARD_TIMEOUT)
            except rest.ApiError as e:
                if e.status != 404:
                    still = " (still running)" if e.status == 409 else ""
                    left.append(f"Soniox kept the transcription{still}: {rest.clip(e, 120)}")
            except OSError as e:
                left.append(f"could not delete the transcription: {rest.clip(e, 120)}")
        if job.extra.get("file_id"):
            try:
                self.request("DELETE", f"/v1/files/{job.extra['file_id']}", timeout=rest.DISCARD_TIMEOUT)
            except rest.ApiError as e:
                if e.status != 404:
                    left.append(f"Soniox kept the file: {rest.clip(e, 120)}")
            except OSError as e:
                left.append(f"could not delete the file: {rest.clip(e, 120)}")
        return "; ".join(left) or None


def to_transcript(tokens):
    """Group Soniox tokens into utterances: a new one whenever the speaker
    changes. Token text carries its own leading space, so joining is plain
    concatenation."""
    letters = {}
    groups = []  # [speaker, [tokens]]
    for token in tokens:
        speaker = token.get("speaker")
        if speaker is not None and speaker not in letters:
            letters[speaker] = speaker_letter(len(letters))
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(token)
        else:
            groups.append([speaker, [token]])

    utterances = []
    for speaker, group in groups:
        text = "".join(t.get("text", "") for t in group).strip()
        if not text:
            continue
        languages = [t["language"] for t in group if t.get("language")]
        utterances.append(Utterance(
            text=text,
            speaker=letters.get(speaker),
            start_ms=group[0].get("start_ms"),
            end_ms=group[-1].get("end_ms"),
            language=max(set(languages), key=languages.count) if languages else None,
        ))
    text = "".join(t.get("text", "") for t in tokens).strip()
    return Transcript(provider=Soniox.name, text=text, utterances=utterances)
