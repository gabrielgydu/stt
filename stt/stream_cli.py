"""
`stt stream`: live transcription for tools that are not Python.

    capture | stt stream [--rate 16000] [--terms T | --terms-file F] [--languages L]
                         [--speakers] [--endpoint-ms N] [--provider P] [--finish-timeout 5]

stdin is raw PCM s16le mono at --rate. It is read from process start, before
connecting, so audio captured while stt connects is kept (a pipe holds only
about 2 s). stdout is one JSON object per line, UTF-8, flushed at once, and
nothing else ever goes there:

    {"type": "ready", "provider": "soniox", "model": "stt-rt-v5"}
    {"type": "note", "message": "..."}
    {"type": "partial", "turn": 0, "text": "...", "start_ms": 120}
    {"type": "final", "turn": 0, "text": "...", "start_ms": 120, "end_ms": 2300,
     "speaker": null, "language": "pt"}
    {"type": "error", "kind": "auth", "fatal": true, "routine": false,
     "retry_after": null, "message": "..."}
    {"type": "end"}

A partial carries the open turn's whole text and replaces the previous one; a
final closes its turn; turns join with one space. Closing stdin means "done":
the provider finalizes, the last finals and `end` follow, exit 0 (if it does
not answer within --finish-timeout, the last turn as last heard and a note say
so). SIGTERM / SIGINT mean "cancel": the provider session closes at once,
nothing more is printed, exit 143 / 130.

A session that fails mid-way first closes its open turn with a final as last
heard. Failures print the error line, then `stt: <message>` as the last stderr
line, and exit 3 (config, checked before any network), 4 (the provider
refused: retrying will not help), 5 (network, server, rate or session limit:
worth another try) or 1 (internal). There is no reconnect inside stt.
"""

import argparse
import json
import os
import signal
import sys
import threading

from . import config as _config
from . import exitcodes, rest

DEFAULT_RATE = 16000
CONNECT_TIMEOUT = 15
# Audio read from stdin but not sent yet; more than this means stt cannot keep up.
BACKLOG_SECONDS = 60
# The main loop looks at stdin, the session and the signal flag this often.
TICK_SECONDS = 0.02
READ_BYTES = 65536
DESCRIPTION = ("Live transcription: raw PCM s16le mono on stdin, one JSON event per line on stdout. "
               "Close stdin to finish; SIGTERM to cancel.")


def _positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, not {value}")
    return value


def _add_arguments(p):
    p.add_argument("--rate", type=_positive_int, default=DEFAULT_RATE,
                   help=f"sample rate of the PCM on stdin (default: {DEFAULT_RATE})")
    terms = p.add_mutually_exclusive_group()
    terms.add_argument("--terms", help="comma-separated key terms, most important first "
                                       "(default: keyterms from the config; '' for none)")
    terms.add_argument("--terms-file", help="key terms, one per line (for terms with commas)")
    p.add_argument("--languages", help="comma-separated language hints "
                                       "(default: languages from the config; '' to auto-detect)")
    p.add_argument("--speakers", action="store_true", help="label speakers (off by default)")
    p.add_argument("--endpoint-ms", type=_positive_int,
                   help="longest silence before a turn ends (default: the provider's)")
    p.add_argument("--provider", help="override the configured provider (soniox, assemblyai, fake)")
    p.add_argument("--finish-timeout", type=float, default=5,
                   help="after stdin ends, wait this long for the last turn (default: 5)")


def add_parser(subparsers):
    p = subparsers.add_parser(
        "stream", help="live transcription: raw PCM on stdin, JSON lines on stdout",
        description=DESCRIPTION,
    )
    _add_arguments(p)
    p.set_defaults(run=run)
    return p


class _Cancel(BaseException):
    """Raised by the signal handler to cut a blocking finish() short."""


class _Signals:
    """SIGTERM / SIGINT set the exit status. The main loop checks it every
    tick. The handler also aborts the session, so a send stuck on a stalled
    network fails at once instead of being retried after the signal; and
    while finish() blocks, it raises."""

    def __init__(self):
        self.status = None
        self.interrupt = False
        self.session = None
        try:
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
        except ValueError:  # not the main thread: the caller handles signals
            pass

    def _on_signal(self, signum, _frame):
        self.status = exitcodes.SIGTERM if signum == signal.SIGTERM else exitcodes.SIGINT
        if self.session is not None:
            self.session.abort()
        if self.interrupt:
            raise _Cancel()


class _Stdin(threading.Thread):
    """Reads stdin into memory from the start, so nothing captured while stt
    connects is lost. Stops reading once more than `limit` bytes wait unsent."""

    def __init__(self, fd, limit):
        super().__init__(name="stt-stdin", daemon=True)
        self.fd, self.limit = fd, limit
        self.lock = threading.Lock()
        self.chunks, self.size = [], 0
        self.eof = self.overflow = False

    def run(self):
        while True:
            try:
                data = os.read(self.fd, READ_BYTES)
            except OSError:
                data = b""
            with self.lock:
                if data:
                    self.chunks.append(data)
                    self.size += len(data)
                    self.overflow = self.size > self.limit
                else:
                    self.eof = True
                if self.eof or self.overflow:
                    return

    def take(self):
        """(audio not taken yet, stdin ended, backlog overflowed)."""
        with self.lock:
            data = b"".join(self.chunks)
            self.chunks, self.size = [], 0
            return data, self.eof, self.overflow


class _Out:
    """JSON lines on stdout, each written whole and flushed at once. Nothing
    is written once a signal came in. A closed stdout never raises here, so
    the error paths still reach stderr and their exit status; `gone` says so
    and the main loop stops on it."""

    def __init__(self, signals):
        self.signals = signals
        self.stream = sys.stdout.buffer
        self.gone = False

    def line(self, obj):
        if self.signals.status is not None or self.gone:
            return
        try:
            self.stream.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.stream.flush()
        except BrokenPipeError:
            self.gone = True
            # Whoever read stdout is gone; keep Python's exit flush from complaining.
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())

    def event(self, event):
        self.line({"type": type(event).__name__.lower(), **vars(event)})


def _connect(stream, signals, kwargs):
    """stream.connect() on a helper thread, so a signal is answered at once
    even mid-handshake. None when cancelled meanwhile (the half-open socket
    goes with the process)."""
    result = {}

    def attempt():
        try:
            result["session"] = stream.connect(**kwargs)
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            result["error"] = e

    thread = threading.Thread(target=attempt, name="stt-connect", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(TICK_SECONDS)
        if signals.status is not None:
            return None
    if "error" in result:
        raise result["error"]
    return result["session"]


def _pump(sess, audio, out, signals, finish_timeout):
    """Audio in, events out, until stdin ends (then finish) or a signal."""
    from .stream import StreamError
    while True:
        if signals.status is not None:
            return signals.status
        if out.gone:
            raise BrokenPipeError("stdout closed")
        data, eof, overflow = audio.take()
        if overflow:
            raise StreamError(f"more than {BACKLOG_SECONDS} s of audio piled up unsent", "internal")
        if data:
            sess.send(data)
        if eof:
            break
        for event in sess.poll(0 if data else TICK_SECONDS):
            out.event(event)
    signals.interrupt = True
    try:
        if signals.status is not None:
            raise _Cancel()
        events = sess.finish(finish_timeout)
    finally:
        signals.interrupt = False
    for event in events:
        out.event(event)
    if sess.timed_out:
        out.line({"type": "note", "message": sess.timed_out})
    out.line({"type": "end"})
    if out.gone:
        raise BrokenPipeError("stdout closed")
    return signals.status or exitcodes.OK


def _status(error):
    if error.kind == "config":
        return exitcodes.CONFIG
    if error.fatal:
        return exitcodes.FATAL
    if error.kind == "internal":
        return exitcodes.INTERNAL
    return exitcodes.TRANSIENT


def run(args):
    """`stt stream`; returns the exit status."""
    from . import stream   # here, so `stt transcribe` does not load it

    signals = _Signals()
    audio = _Stdin(0, BACKLOG_SECONDS * 2 * args.rate)
    audio.start()
    out = _Out(signals)
    secrets = []
    sess = None

    def fail(error):
        message = rest.clip(rest.scrub(str(error), secrets))
        out.line({"type": "error", "kind": error.kind, "fatal": error.fatal, "routine": error.routine,
                  "retry_after": error.retry_after, "message": message})
        if signals.status is None:
            print(f"stt: {message}", file=sys.stderr, flush=True)
        return signals.status or _status(error)

    try:
        cfg = stream._load_config()
        secrets = _config.api_keys(cfg)
        if args.terms_file is not None:
            try:
                terms = _config.read_terms_file(args.terms_file)
            except (OSError, UnicodeDecodeError) as e:
                reason = getattr(e, "strerror", None) or e
                raise stream.StreamError(f"--terms-file {args.terms_file}: {reason}", "config") from None
        else:
            terms = None if args.terms is None else _config.split_list(args.terms)
        languages = None if args.languages is None else _config.split_list(args.languages)
        problem = stream.config_problem(cfg, args.provider)
        if problem:
            raise stream.StreamError(problem, "config")
        sess = _connect(stream, signals, dict(
            sample_rate=args.rate, terms=terms, languages=languages, speakers=args.speakers,
            endpoint_ms=args.endpoint_ms, config=cfg, provider_name=args.provider,
            timeout=CONNECT_TIMEOUT,
        ))
        if sess is None:
            return signals.status
        signals.session = sess
        info = sess.info
        out.line({"type": "ready", "provider": info.provider, "model": info.model})
        if info.note:
            out.line({"type": "note", "message": rest.clip(info.note)})
        return _pump(sess, audio, out, signals, args.finish_timeout)
    except _Cancel:
        return signals.status
    except stream.StreamError as e:
        return fail(e)
    except stream.SessionClosed as e:
        for event in e.events:
            out.event(event)
        return fail(e.error or stream.StreamError("the session ended unexpectedly", "internal"))
    except BrokenPipeError:   # from _pump, once _Out found stdout closed
        print("stt: stdout closed", file=sys.stderr, flush=True)
        return signals.status or exitcodes.INTERNAL
    except Exception as e:  # noqa: BLE001 - every failure ends as one error line
        if os.environ.get("STT_DEBUG") == "1":
            import traceback
            traceback.print_exc()
        return fail(stream.StreamError(f"internal error: {type(e).__name__}: {e}", "internal"))
    finally:
        if sess is not None:
            sess.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stt stream", description=DESCRIPTION)
    _add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
