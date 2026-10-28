"""
The `stt` command, for tools that are not Python.

    stt transcribe FILE|- [--json] [--no-speakers] [-o OUT] [--timeout S]
                   [--terms T | --terms-file F] [--languages L] [--spellings F]
                   [--provider P] [--keep-on-timeout]
    stt check [--json] [--provider P]
    stt stream ...                      (live transcription, see stt.stream_cli)

`transcribe` writes the transcript to stdout (or OUT) and progress to stderr.
FILE `-` reads the audio from stdin into memory; nothing goes to disk. Plain
output is the "Speaker A: ..." format; --json is {"provider", "text",
"utterances": [{"text", "speaker", "start_ms", "end_ms", "language"}]}. The
transcript is written and flushed before the provider-side copy is deleted,
so cleanup never delays it.

`check` prints the configured provider and whether live transcription can
run, without touching the network; --json prints the same as one object.

Exit statuses (stt.exitcodes): 0 ok; 1 internal error, or the transcript
could not be written (closed stdout, full disk); 2 bad arguments; 3 config
problem; 4 the provider refused (bad key, no credit, rejected or too long
audio, failed job); 5 transient (network, 5xx, rate limit) after retries;
6 gave up waiting; 130/143 interrupted by SIGINT/SIGTERM, the job deleted on
the way out. On failure the last stderr line is "stt: <message>": one line of
at most ~300 characters, every configured API key replaced by ***. Progress
lines are scrubbed the same way.
"""

import argparse
import json
import os
import signal
import sys

from . import ConfigError, JobFailed, StillRunning, config, exitcodes, provider, providers, rest, transcribe
from . import stream_cli  # light: it loads stt.stream only when `stt stream` runs


class UsageError(Exception):
    """A bad argument found after parsing: missing file, unreadable list."""


class OutputError(Exception):
    """The transcript could not be written: stdout closed, disk full."""


class Cancelled(BaseException):
    """SIGTERM or SIGINT. A BaseException like KeyboardInterrupt, so it unwinds
    through stt.transcribe's cleanup and the job is deleted on the way out."""

    def __init__(self, status):
        super().__init__(status)
        self.status = status


# The first SIGTERM/SIGINT cancels; later ones are ignored so the cleanup it
# started can finish. Once the transcript is out only cleanup is left, so a
# signal then just sets the exit status.
_signals = {"status": None, "delivered": False}


def _on_signal(signum, _frame):
    if _signals["status"] is not None:
        return
    _signals["status"] = exitcodes.SIGTERM if signum == signal.SIGTERM else exitcodes.SIGINT
    if not _signals["delivered"]:
        raise Cancelled(_signals["status"])


def _signal_name(status):
    return "SIGTERM" if status == exitcodes.SIGTERM else "SIGINT"


# The configured API keys, once the config is read: every stderr line is
# scrubbed, progress included (a provider may echo the key in an error that a
# retry line repeats).
_secrets = []


def _err(message):
    print(rest.scrub(message, _secrets), file=sys.stderr, flush=True)


def _fail(message, notes=()):
    """The last stderr line of a failed command."""
    _err("stt: " + rest.clip(rest.scrub("; ".join([message, *notes]), _secrets)))


def _drop_stdout():
    """Point fd 1 at /dev/null after a write to it failed, so the flush
    Python retries at exit cannot fail again and turn the exit status into
    120 with a traceback after our last line."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except (OSError, ValueError):
        pass


def _stdout(text):
    """Write `text` to stdout and flush it; OutputError when that fails."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except OSError as e:
        _drop_stdout()
        raise OutputError(f"cannot write to stdout: {e.strerror or e}") from None


class _Parser(argparse.ArgumentParser):
    """argparse, with its error as the usual last line "stt: <message>"."""

    def error(self, message):
        self.print_usage(sys.stderr)
        _err(f"stt: {message}")
        sys.exit(exitcodes.USAGE)


def _load_config():
    try:
        return config.load()
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {config.path()}: {getattr(e, 'strerror', None) or e}") from None


def _status(error):
    """The exit status for an exception that ended a command."""
    if isinstance(error, UsageError):
        return exitcodes.USAGE
    if isinstance(error, OutputError):
        return exitcodes.INTERNAL
    if isinstance(error, ConfigError):
        return exitcodes.CONFIG
    if isinstance(error, StillRunning):
        return exitcodes.TIMEOUT
    if isinstance(error, JobFailed):  # TooLong included
        return exitcodes.FATAL
    if isinstance(error, rest.ApiError):
        return exitcodes.TRANSIENT if error.transient else exitcodes.FATAL
    if rest.is_transient(error):
        return exitcodes.TRANSIENT
    return exitcodes.INTERNAL


def _terms(args):
    if args.terms_file is not None:
        try:
            return config.read_terms_file(args.terms_file)
        except (OSError, UnicodeDecodeError) as e:
            raise UsageError(f"--terms-file {args.terms_file}: {getattr(e, 'strerror', None) or e}") from None
    if args.terms is not None:
        return config.split_list(args.terms)
    return None


def _spellings(args):
    if not args.spellings:
        return None
    from . import spelling
    try:
        return spelling.load(args.spellings)
    except (OSError, ValueError) as e:
        raise UsageError(f"--spellings {args.spellings}: {getattr(e, 'strerror', None) or e}") from None


def _check_output(output):
    """UsageError unless the transcript can be written to `output`: caught
    here, not after the provider has transcribed (and billed) the audio."""
    folder = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(folder):
        raise UsageError(f"-o {output}: no such directory")
    if os.path.isdir(output):
        raise UsageError(f"-o {output}: is a directory")
    if os.path.exists(output):
        if not os.access(output, os.W_OK):
            raise UsageError(f"-o {output}: not writable")
    elif not os.access(folder, os.W_OK | os.X_OK):
        raise UsageError(f"-o {output}: cannot create files in {folder}")


def cmd_transcribe(args, cfg):
    terms = _terms(args)
    languages = config.split_list(args.languages) if args.languages is not None else None
    spellings = _spellings(args)
    if args.file != "-":
        # Any readable non-directory: a pipe (<(...), a FIFO) works too.
        if not os.path.exists(args.file):
            raise UsageError(f"no such file: {args.file}")
        if os.path.isdir(args.file):
            raise UsageError(f"{args.file} is a directory")
        if not os.access(args.file, os.R_OK):
            raise UsageError(f"cannot read {args.file}")
    if args.output:
        _check_output(args.output)
    provider(cfg, args.provider)  # a config problem fails before stdin is read

    if args.file == "-":
        audio = sys.stdin.buffer.read()
        if not audio:
            raise UsageError("no audio on stdin")
    else:
        audio = args.file

    def deliver(transcript):
        if args.json:
            out = json.dumps(transcript.to_dict(), ensure_ascii=False)
        else:
            out = transcript.format()
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(out + ("\n" if args.json else ""))
            except OSError as e:
                raise OutputError(f"cannot write {args.output}: {e.strerror or e}") from None
        else:
            _stdout(out + "\n")
        _signals["delivered"] = True
        if args.output:
            _err(f"Transcript saved to: {args.output}")

    transcribe(
        audio,
        speakers=not args.no_speakers,
        terms=terms,
        languages=languages,
        timeout=args.timeout,
        progress=_err,
        config=cfg,
        provider_name=args.provider,
        spellings=spellings,
        deliver=deliver,
        keep_on_timeout=args.keep_on_timeout,
    )
    return _signals["status"] or exitcodes.OK


def _stream_status(cfg, provider_name):
    """(problem, model) for live transcription; problem None = ready."""
    try:
        from . import stream
    except ImportError:
        return "not available", None
    try:
        problem = stream.config_problem(config=cfg, provider_name=provider_name)
    except Exception as e:  # noqa: BLE001 - check reports, it does not crash
        return rest.clip(f"check failed: {type(e).__name__}: {e}", 200), None
    try:
        model = stream.model_name(config=cfg, provider_name=provider_name)
    except Exception as e:  # noqa: BLE001
        model = None
        problem = problem or rest.clip(f"no model: {type(e).__name__}: {e}", 200)
    return problem, model


def cmd_check(args, cfg):
    name = args.provider or config.provider_name(cfg)
    problem, limit = None, None
    try:
        limit = provider(cfg, args.provider).max_seconds
    except ConfigError as e:
        problem = str(e)
        try:
            limit = providers.get(name).max_seconds
        except ConfigError:
            pass
    stream_problem, model = _stream_status(cfg, args.provider)
    if args.json:
        _stdout(json.dumps({
            "provider": name,
            "config": config.path(),
            "ok": problem is None,
            "problem": problem,
            "max_seconds": limit,
            "stream": {"ok": stream_problem is None, "problem": stream_problem, "model": model},
        }, ensure_ascii=False) + "\n")
    else:
        _stdout(f"provider: {name} (config: {config.path()})\n"
                + (f"stream: ready ({model})" if stream_problem is None else f"stream: {stream_problem}") + "\n")
    if problem:
        raise ConfigError(problem)
    return exitcodes.OK


def main(argv=None):
    parser = _Parser(prog="stt", description="Speech-to-text through the configured provider.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe an audio or video file")
    t.add_argument("file", help="the audio file, or - to read it from stdin")
    t.add_argument("-o", "--output", help="write the transcript here instead of stdout")
    t.add_argument("--json", action="store_true", help="JSON with utterances and timestamps")
    t.add_argument("--no-speakers", action="store_true", help="no speaker separation")
    terms = t.add_mutually_exclusive_group()
    terms.add_argument("--terms", help="comma-separated key terms (default: keyterms from the config)")
    terms.add_argument("--terms-file", metavar="PATH", help="key terms, one per line (they may contain commas)")
    t.add_argument("--languages", help="comma-separated language hints (default: languages from the config)")
    t.add_argument("--spellings", metavar="FILE",
                   help='JSON [{"from": [...], "to": "..."}] applied to the result')
    t.add_argument("--provider", choices=providers.names(), help="override the configured provider")
    t.add_argument("--timeout", type=float, default=3600, help="give up waiting after this many seconds (default: 3600)")
    t.add_argument("--keep-on-timeout", action="store_true",
                   help="on timeout, leave the job at the provider (default: delete it)")
    t.set_defaults(run=cmd_transcribe)

    c = sub.add_parser("check", help="show the configured provider or what is wrong with the config")
    c.add_argument("--json", action="store_true", help="the same as one JSON object")
    c.add_argument("--provider", choices=providers.names())
    c.set_defaults(run=cmd_check)

    stream_cli.add_parser(sub)

    args = parser.parse_args(argv)
    if args.command == "stream":
        return stream_cli.run(args)

    _signals.update(status=None, delivered=False)
    previous = {}
    if args.command == "transcribe":
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, _on_signal)
        except ValueError:  # not the main thread: no handlers, Ctrl-C as usual
            pass
    _secrets.clear()
    try:
        cfg = _load_config()
        _secrets.extend(config.api_keys(cfg))
        status = args.run(args, cfg)
        if status in (exitcodes.SIGTERM, exitcodes.SIGINT):
            # The signal came after the transcript was out; cleanup finished.
            _fail(f"{_signal_name(status)} arrived after the transcript was written")
        return status
    except Cancelled as e:
        # The job was deleted on the way out; the notes name what was not.
        _fail(f"cancelled by {_signal_name(e.status)}", getattr(e, "__notes__", ()))
        return e.status
    except KeyboardInterrupt as e:
        _fail("cancelled by SIGINT", getattr(e, "__notes__", ()))
        return exitcodes.SIGINT
    except Exception as e:  # noqa: BLE001 - every failure ends as one "stt:" line
        status = _status(e)
        if status == exitcodes.INTERNAL and not isinstance(e, OutputError):
            if os.environ.get("STT_DEBUG") == "1":
                import traceback
                _err(traceback.format_exc().rstrip())  # scrubbed like every line
            message = f"internal error: {type(e).__name__}: {e}"
        else:
            message = str(e)
        _fail(message)
        return status
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
