"""
Speech-to-text providers behind one interface.

A provider transcribes a file as a job: `submit` uploads and queues it,
`status` returns the Transcript once it is done (None while it runs), and
`discard` deletes whatever the job left on the provider's side (`cleanup` is
the same, finished even when Ctrl-C cuts into it). The caller owns the wait
loop (see stt.wait), which is what lets the batch tool keep many jobs in
flight at once.

The audio given to `submit` is a file path, or the audio itself as bytes
(`stt transcribe -` reads stdin into memory, so nothing touches the disk).

Adding a provider = one module here with a Provider subclass, plus its entry
in _registry().
"""

import os
from dataclasses import dataclass, field

from .. import rest


class ConfigError(RuntimeError):
    """The config does not allow transcribing: unknown provider, missing key."""


class JobFailed(RuntimeError):
    """The provider ran the job and reported it as failed. Not worth retrying."""


class TooLong(JobFailed):
    """The audio is longer than the provider accepts; refused before uploading.
    A JobFailed, since sending it again cannot help either."""


class LeftBehind(ConnectionError):
    """A submit failed on the network or with a 5xx and may have left
    something at the provider that stt cannot delete: the answer to creating
    the job was lost, or deleting the partial upload failed as well. Transient
    like the failure behind it, but stt.submit does not try again, since every
    attempt could leave one more behind (and a second job is billed twice)."""


@dataclass
class Job:
    id: str
    path: str
    # Provider-specific leftovers to clean up, e.g. Soniox's uploaded file id.
    extra: dict = field(default_factory=dict)


def audio_name(audio):
    """A name for the audio in uploads and messages: the file's basename, or
    "audio" for bytes."""
    if isinstance(audio, (bytes, bytearray)):
        return "audio"
    return os.path.basename(audio)


def audio_path(audio):
    """What Job.path holds: the file path, or "-" for audio given as bytes."""
    if isinstance(audio, (bytes, bytearray)):
        return "-"
    return os.fspath(audio)


def read_audio(audio):
    """The audio as bytes, whether given as bytes or as a path."""
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    with open(audio, "rb") as f:
        return f.read()


class Provider:
    name = ""
    # Longest audio the provider accepts in one file.
    max_seconds = 0
    # Seconds between status checks (stt.wait polls faster at first).
    poll_seconds = 5
    # Pauses before retrying a submit that failed on the network or with a
    # 5xx/429; one retry per entry.
    submit_retry_waits = (2, 5)
    needs_key = True

    def __init__(self, config, config_path):
        self.config = config
        key_name = f"{self.name}_api_key"
        self.api_key = config.get(key_name)
        if self.needs_key and not self.api_key:
            raise ConfigError(f"{key_name} is not set in {config_path}")

    def submit(self, audio, *, speakers, terms, languages, progress):
        raise NotImplementedError

    def status(self, job):
        raise NotImplementedError

    def discard(self, job):
        """Delete the job's audio and transcript on the provider's side.

        Returns None when nothing is left there, else a short note on what is
        left and why. Never raises: cleanup failing must not lose a transcript
        already fetched.
        """
        return None

    def cleanup(self, job):
        """discard(job), finished even when Ctrl-C or SIGTERM cuts into it.

        An interruption (the `stt` command turns the first SIGTERM into one)
        would otherwise stop between two DELETEs and leave the rest behind.
        The interrupted discard runs once more (deleting is idempotent; 404
        counts as gone), then the interruption goes on, carrying a note when
        something is still left. Returns what discard returns.
        """
        try:
            return self.discard(job)
        except BaseException as e:
            left = self.discard(job)
            if left:
                e.add_note(f"could not clean up {self.name} job {job.id or '(no id yet)'}: {left}")
            raise

    def scrub(self, text):
        """`text` on one short line without this provider's key, for job
        error messages that echo the request."""
        return rest.clip(rest.scrub(text, (self.api_key,)))

    def _headers(self):
        return {}

    def request(self, method, route, **kwargs):
        """rest.request against this provider, with its auth header and with
        the key scrubbed from any error message."""
        headers = {**self._headers(), **kwargs.pop("headers", {})}
        return rest.request(method, self.base_url + route, headers=headers,
                            secrets=(self.api_key,), **kwargs)


def _registry():
    from .assemblyai import AssemblyAI
    from .fake import Fake
    from .soniox import Soniox
    return {p.name: p for p in (Soniox, AssemblyAI, Fake)}


def names():
    return sorted(_registry())


def get(name):
    """The Provider class called `name`, or ConfigError."""
    try:
        return _registry()[name]
    except KeyError:
        raise ConfigError(
            f"unknown provider {name!r}; known: {', '.join(names())}"
        ) from None
