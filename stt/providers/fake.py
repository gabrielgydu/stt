"""
A provider that never touches the network, for the tests of tools that use
stt: `provider=fake` in a throwaway config (STT_CONFIG) drives the real `stt`
command and Python API without a key, credit or mock server.

The transcript is deterministic: "fake transcript of <name> (<bytes> bytes)"
as one utterance, speaker "A" when speakers are on, language "en".

Config knobs (optional):

    fake_fail = submit:<kind>    submitting fails like a provider would; kind is
                                 config | auth | credit | rejected | rate_limit |
                                 network | server | limit | timeout | internal
    fake_fail = job              the job runs and fails (JobFailed)
    fake_fail = slow:<seconds>   the job takes that long, to test timeouts

A transient submit failure is retried as for real providers, with 0.1 s
pauses so consumer tests stay fast; it fails every attempt, so the command
ends with exit 5. The streaming fake (stt.stream) reads the same key; its
values (connect:..., after:...) are ignored here.
"""

import time

from ..rest import ApiError
from ..transcript import Transcript, Utterance
from . import ConfigError, Job, JobFailed, Provider, TooLong, audio_name, audio_path, read_audio

# What a refused submit looks like for each kind: (HTTP status, reason).
_HTTP = {
    "auth": (401, "Unauthorized"),
    "credit": (402, "Payment Required"),
    "rejected": (400, "Bad Request"),
    "rate_limit": (429, "Too Many Requests"),
    "server": (503, "Service Unavailable"),
}
KINDS = sorted([*_HTTP, "config", "network", "timeout", "limit", "internal"])


class Fake(Provider):
    name = "fake"
    max_seconds = 300 * 60
    poll_seconds = 0.1
    submit_retry_waits = (0.1, 0.1)
    needs_key = False

    def __init__(self, config, config_path):
        super().__init__(config, config_path)
        self.fail_submit = None
        self.fail_job = False
        self.slow = 0.0
        knob = (config.get("fake_fail") or "").strip()
        what, _, arg = knob.partition(":")
        if not knob or what in ("connect", "after"):
            pass
        elif what == "submit" and arg in KINDS:
            self.fail_submit = arg
        elif knob == "job":
            self.fail_job = True
        elif what == "slow":
            try:
                self.slow = float(arg)
            except ValueError:
                raise ConfigError(f"fake_fail = {knob}: slow needs seconds, e.g. slow:5") from None
        else:
            raise ConfigError(
                f"fake_fail = {knob} in {config_path}: expected submit:<kind>, job or slow:<seconds>"
                f" (kinds: {', '.join(KINDS)})"
            )

    def submit(self, audio, *, speakers, terms, languages, progress):
        progress("Uploading audio to the fake provider...")
        kind = self.fail_submit
        if kind in _HTTP:
            status, reason = _HTTP[kind]
            raise ApiError(status, f"HTTP {status} {reason}: fake {kind} failure")
        if kind == "config":
            raise ConfigError("fake config failure")
        if kind == "network":
            raise ConnectionError("fake network failure")
        if kind == "timeout":
            raise TimeoutError("fake upload timed out")
        if kind == "limit":
            raise TooLong("fake: audio longer than the provider accepts")
        if kind == "internal":
            raise ValueError("fake internal failure")
        size = len(read_audio(audio))
        return Job(id=f"fake-{time.monotonic_ns()}", path=audio_path(audio), extra={
            "text": f"fake transcript of {audio_name(audio)} ({size} bytes)",
            "speakers": bool(speakers),
            "ready_at": time.monotonic() + self.slow,
        })

    def status(self, job):
        if self.fail_job:
            raise JobFailed("fake job failed")
        if time.monotonic() < job.extra["ready_at"]:
            return None
        text = job.extra["text"]
        return Transcript(provider=self.name, text=text, utterances=[Utterance(
            text=text,
            speaker="A" if job.extra["speakers"] else None,
            start_ms=0,
            language="en",
        )])
