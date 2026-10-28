"""
stt — one speech-to-text interface for every transcription tool.

    import stt
    problem = stt.config_problem()        # None when transcription can run
    transcript = stt.transcribe("rec.mp3", progress=print)
    text = transcript.format()            # "Speaker A: ..." blocks

The provider, its key, the language hints and the key terms all come from
~/.config/stt/config (see stt.config). Tools that are not Python run the `stt`
command instead (see stt.cli). Live transcription is stt.stream, which this
module does not import, so `import stt` never needs websockets.
"""

import os
import time

from . import config as _config
from . import rest
from .providers import ConfigError, Job, JobFailed, LeftBehind, Provider, TooLong
from .providers import get as _provider_class
from .providers import read_audio
from .transcript import Transcript, Utterance

__all__ = [
    "ConfigError", "Job", "JobFailed", "LeftBehind", "Provider", "StillRunning", "TooLong",
    "Transcript", "Utterance", "config_problem", "max_seconds", "provider", "submit", "transcribe", "wait",
]

# Give up on a job (StillRunning) after this many seconds without a successful
# status check. Time, not a count: a Wi-Fi roam or a DNS hiccup fails every
# check at once, and the job keeps running server-side meanwhile.
POLL_ERROR_WINDOW = 60

# Short clips are done in seconds, so the first checks come quickly; after
# that the provider's own poll_seconds applies.
FAST_POLL_SECONDS = 1
FAST_POLL_WINDOW = 20

# ffprobe gets this long to read a file's duration before the guard gives up.
PROBE_TIMEOUT = 30


class StillRunning(RuntimeError):
    """Gave up waiting for a job (timeout, or status checks kept failing).

    `wait` leaves the job alone, so it may still finish; `transcribe` deletes
    it unless told to keep it. The message says which happened; `reason` is
    why stt gave up."""

    def __init__(self, message, reason=None):
        super().__init__(message)
        self.reason = reason or message


def provider(config=None, name=None):
    """The configured Provider, ready to use. Raises ConfigError."""
    config_path = _config.path()
    config = _config.load(config_path) if config is None else config
    cls = _provider_class(name or _config.provider_name(config))
    return cls(config, config_path)


def config_problem(config=None):
    """Why transcription cannot run with the current config, or None."""
    try:
        provider(config)
    except ConfigError as e:
        return str(e)
    return None


def max_seconds(config=None):
    """Longest audio the configured provider accepts in one file."""
    return provider(config).max_seconds


def _quiet(_message):
    pass


def _duration(path):
    """Seconds of audio in the file by ffprobe, or None when ffprobe is missing
    or cannot tell."""
    import shutil
    import subprocess  # here, so `import stt` stays light for `stt stream`
    if shutil.which("ffprobe") is None:
        return None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", "file:" + os.path.abspath(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT,
        )
        return float(probe.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _readable(seconds):
    """ " (5.0 h)" or " (12 min)" after a count of seconds; nothing under a minute."""
    if seconds >= 3600:
        return f" ({seconds / 3600:.1f} h)"
    return f" ({seconds / 60:.0f} min)" if seconds >= 60 else ""


def submit(prov, audio, *, speakers, terms, languages, progress=_quiet):
    """prov.submit with two things every caller wants: a file longer than the
    provider accepts is refused before uploading (TooLong, when ffprobe can
    tell), and a submit that failed on the network or with a 5xx/429 is tried
    again after each of prov.submit_retry_waits. `audio` is a path or bytes.

    Not tried again: LeftBehind, a failure that may have left a job or an
    upload at the provider that stt cannot delete (another attempt could
    leave one more, and a second job is billed twice)."""
    if not isinstance(audio, (bytes, bytearray)) and not os.path.isfile(audio):
        # A pipe (FIFO, /dev/stdin, <(...)) can be read only once: read it
        # now, so neither ffprobe nor a retry finds it empty.
        audio = read_audio(audio)
    if not isinstance(audio, (bytes, bytearray)):
        seconds = _duration(audio)
        if seconds is not None and seconds > prov.max_seconds:
            raise TooLong(
                f"the audio is {seconds:.0f} s{_readable(seconds)}; {prov.name} accepts at most "
                f"{prov.max_seconds:.0f} s{_readable(prov.max_seconds)}. Nothing was uploaded."
            )
    waits = list(prov.submit_retry_waits)
    while True:
        try:
            return prov.submit(audio, speakers=speakers, terms=terms,
                               languages=languages, progress=progress)
        except (rest.ApiError, OSError) as e:
            if not waits or not rest.is_transient(e) or isinstance(e, LeftBehind):
                raise
            pause = waits.pop(0)
            progress(f"Submitting to {prov.name} failed ({rest.clip(e, 120)}); retrying in {pause:g}s...")
            time.sleep(pause)


def wait(prov, job, timeout, progress=_quiet):
    """Poll `job` until it finishes and return its Transcript.

    Raises JobFailed when the provider reports a failure, and StillRunning on
    timeout or when status checks have failed for POLL_ERROR_WINDOW seconds
    (naming the job id; the job is left in place).
    """
    started = time.monotonic()
    deadline = started + timeout
    last_ok = started
    next_note = 30
    while True:
        now = time.monotonic()
        pause = prov.poll_seconds
        if now - started < FAST_POLL_WINDOW:
            pause = min(pause, FAST_POLL_SECONDS)
        time.sleep(max(0, min(pause, deadline - now)))
        try:
            result = prov.status(job)
            last_ok = time.monotonic()
        except JobFailed:
            raise
        except (rest.ApiError, OSError) as e:
            if isinstance(e, rest.ApiError) and not e.transient:
                raise
            failing = time.monotonic() - last_ok
            if failing >= POLL_ERROR_WINDOW:
                reason = f"status checks failed for {failing:.0f}s (last: {rest.clip(e, 100)})"
                raise StillRunning(f"{reason}; {prov.name} job {job.id} may still finish.", reason) from None
            result = None
        if result is not None:
            return result
        elapsed = time.monotonic() - started
        if elapsed >= next_note:
            progress(f"Still transcribing ({int(elapsed)}s)...")
            next_note += 30
        if time.monotonic() >= deadline:
            reason = f"transcription timed out after {timeout:g}s"
            raise StillRunning(f"{reason}; {prov.name} job {job.id} may still finish.", reason)


def transcribe(path, *, speakers=True, terms=None, languages=None, timeout=600,
               progress=_quiet, config=None, provider_name=None, spellings=None,
               deliver=None, keep_on_timeout=False):
    """Transcribe one audio file with the configured provider.

    `path` is a file path, or the audio itself as bytes. `terms` and
    `languages` default to the config's keyterms and languages; pass a list
    (even an empty one) to override. `spellings` is a list of
    {"from": [...], "to": "..."} rules applied to the result (see
    stt.spelling).

    `deliver(transcript)`, when given, is called with the finished transcript
    before the provider-side copy is deleted, so the result is out before any
    cleanup can hold it up. The copy is deleted after every outcome; on
    StillRunning too unless `keep_on_timeout`, and the message says whether
    the job was deleted or may still finish. A cleanup that a
    KeyboardInterrupt (or the command's SIGTERM) cuts into is finished first
    (see Provider.cleanup); what it could not delete is added to the
    exception as a note.
    """
    config = _config.load() if config is None else config
    prov = provider(config, provider_name)
    rules = None
    if spellings:
        from .spelling import Spellings
        rules = Spellings(spellings)  # a bad rule fails here, before any upload
    job = submit(
        prov, path,
        speakers=speakers,
        terms=_config.keyterms(config) if terms is None else terms,
        languages=_config.languages(config) if languages is None else languages,
        progress=progress,
    )
    try:
        progress(f"Transcribing with {prov.name}...")
        transcript = wait(prov, job, timeout, progress)
        if rules:
            rules.apply_transcript(transcript)
        if deliver is not None:
            deliver(transcript)
    except StillRunning as e:
        if keep_on_timeout:
            raise
        left = prov.cleanup(job)
        if left:
            message = f"{e.reason}; {prov.name} job {job.id} may still finish ({left})."
        else:
            message = f"{e.reason}; the {prov.name} job was deleted."
        raise StillRunning(message, e.reason) from None
    except BaseException as e:
        left = prov.cleanup(job)
        if left:
            note = f"Could not clean up {prov.name} job {job.id}: {left}"
            progress(note)
            e.add_note(note)
        raise
    left = prov.cleanup(job)
    if left:
        progress(f"Could not clean up {prov.name} job {job.id}: {left}")
    progress("Transcription done.")
    return transcript
