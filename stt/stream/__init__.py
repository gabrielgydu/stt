"""
Live transcription: raw PCM in, turns of text out, over one provider
connection.

    from stt import stream
    problem = stream.config_problem()       # None when a session can start
    sess = stream.connect(sample_rate=16000)
    sess.send(pcm)                           # s16le mono, any length
    for event in sess.poll(0.1):             # Partial / Final, in order
        ...
    events = sess.finish()                   # end of audio: the open turn comes back final
    sess.close()
    sess.abort()                             # from a signal handler or another thread

Why it is shaped like this:

- One connection, one attempt, no hidden reconnect. Providers bill per
  connected second, and callers differ on what a drop should mean
  (audiorecorder reconnects and replays, dictation gives up), so resilience
  lives with them, built on StreamError's fatal / routine / retry_after.
- Turns are provider-neutral. A Partial carries the open turn's whole current
  text, exactly one Final closes each turn, and turns are numbered 0, 1, 2...
  Soniox streams tokens and AssemblyAI streams turns; each provider module
  turns its frames into these same events.
- Times are milliseconds of audio sent into this session. stt never drops or
  inserts audio mid-session, so a caller can map them onto its own clock.
- A dead network must not hang anyone. Websockets' pings catch an idle one; a
  send stuck on a full buffer holds the connection's lock so no ping goes out,
  which is why the socket also gets TCP_USER_TIMEOUT, finish() a watchdog, and
  abort() exists for a caller that has to cut the line from outside.
- `import stt` must work without python-websockets (the phone recorder service
  imports audiorecorder, which imports stt), so neither this package nor its
  provider modules import it at load time; connect() does.

Providers: soniox and assemblyai (realtime APIs), and fake (no network, no
key: deterministic events for consumer tests, see stt.stream.fake).
"""

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass

from .. import config as _config
from .. import rest
from ..transcript import speaker_letter

__all__ = [
    "Final", "Partial", "Session", "SessionClosed", "SessionInfo", "StreamError",
    "config_problem", "connect", "model_name",
]

PROVIDERS = ("assemblyai", "fake", "soniox")

# No audio sent for this long: the provider's keepalive goes out instead
# (Soniox ends a stream after 20 s with neither).
KEEPALIVE_SECONDS = 10
# How long ending a session waits for the provider to answer the close frame
# before dropping the socket, so a cancel stays quick on a stalled network.
CLOSE_SECONDS = 0.5
# Longest wait for a session's threads when it ends.
JOIN_SECONDS = 2
# Websockets' keepalive pings: one every 20 s, and no pong within 20 s means
# the connection is dead (the library's defaults).
PING_SECONDS = 20
# Audio the network has not acknowledged for this long fails the connection
# (TCP_USER_TIMEOUT, Linux). The pings cannot catch that case: a send blocked
# on a full buffer holds the connection's lock, so no ping goes out meanwhile.
STALL_SECONDS = 20

FATAL_KINDS = ("config", "auth", "credit", "rejected")

WEBSOCKETS_MISSING = "live transcription needs python-websockets (pip install websockets, or your distribution's python-websockets)"


# --------------------------------------------------------------------------
# events and errors
# --------------------------------------------------------------------------

@dataclass
class Partial:
    """The open turn's whole current text; it may still change."""
    turn: int
    text: str
    start_ms: int | None = None


@dataclass
class Final:
    """A turn's last word: its whole text. The turn number never comes back."""
    turn: int
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | None = None   # "A", "B", ... only with speakers=True
    language: str | None = None  # ISO code, when the provider reports one


@dataclass
class SessionInfo:
    provider: str
    model: str
    note: str | None             # what stt changed in the request, e.g. key terms cut
    max_burst_seconds: float     # backlog a caller may send faster than real time
    max_session_seconds: float   # the provider ends a session after this long


class StreamError(RuntimeError):
    """A session that could not start, or ended under the caller.

    `kind` is one of config, auth, credit, rejected, rate_limit, network,
    server, limit, timeout, internal. `fatal`: retrying unchanged cannot help.
    `routine`: the provider's maximum session length was reached, open the next
    session quietly. `retry_after`: seconds to wait before reconnecting.
    str(e) is the provider's own message when it sent one.
    """

    def __init__(self, message, kind, code=None, retry_after=None):
        super().__init__(message)
        self.kind = kind
        self.code = code
        self.retry_after = retry_after
        self.fatal = kind in FATAL_KINDS
        self.routine = kind == "limit"


class SessionClosed(Exception):
    """The session has ended. `error` is None after finish() or close(), else
    the StreamError that ended it; `events` are those that arrived before the
    end and were not returned yet."""

    def __init__(self, error=None, events=None):
        super().__init__(str(error) if error else "session closed")
        self.error = error
        self.events = events or []


# --------------------------------------------------------------------------
# the configured provider
# --------------------------------------------------------------------------

def _session_class(name):
    if name == "soniox":
        from .soniox import Soniox
        return Soniox
    if name == "assemblyai":
        from .assemblyai import AssemblyAI
        return AssemblyAI
    if name == "fake":
        from .fake import Fake
        return Fake
    return None


def _load_config():
    """The config file (stt.config.load), with an unreadable one as
    StreamError (config) rather than an OSError."""
    try:
        return _config.load()
    except (OSError, UnicodeDecodeError) as e:
        reason = getattr(e, "strerror", None) or e
        raise StreamError(rest.clip(f"cannot read {_config.path()}: {reason}"), "config") from None


def _resolve(config, provider_name):
    config = _load_config() if config is None else config
    name = (provider_name or _config.provider_name(config)).strip().lower()
    return config, name, _session_class(name)


def _unknown(name):
    return f"unknown provider {name!r}; known for streaming: {', '.join(PROVIDERS)}"


def config_problem(config=None, provider_name=None):
    """Why a live session cannot start with this config, or None. No network:
    unreadable config, unknown provider, missing key, python-websockets not
    installed."""
    try:
        config, name, cls = _resolve(config, provider_name)
    except StreamError as e:
        return str(e)
    if cls is None:
        return _unknown(name)
    return cls.problem(config)


def model_name(config=None, provider_name=None):
    """The realtime model connect() would use. StreamError (config) for an
    unknown provider."""
    config, name, cls = _resolve(config, provider_name)
    if cls is None:
        raise StreamError(_unknown(name), "config")
    return cls.model_for(config)


def connect(sample_rate=16000, terms=None, languages=None, speakers=False, endpoint_ms=None,
            config=None, provider_name=None, timeout=15):
    """Open a live session and wait until the provider has accepted it.

    The audio is raw PCM s16le mono at `sample_rate`. `terms` and `languages`
    default to the config's keyterms and languages; pass a list (even an empty
    one) to override. Terms are in priority order: a provider cap keeps the
    first ones and says so in info.note. `endpoint_ms` is the longest silence
    before a turn ends (None = the provider's default).

    Exactly one attempt. Raises StreamError, never a library exception.
    """
    config, name, cls = _resolve(config, provider_name)
    problem = _unknown(name) if cls is None else cls.problem(config)
    if problem:
        raise StreamError(rest.clip(rest.scrub(problem, _config.api_keys(config))), "config")
    terms = _config.keyterms(config) if terms is None else list(terms)
    languages = _config.languages(config) if languages is None else list(languages)
    sess = cls(config, int(sample_rate), terms, languages, bool(speakers), endpoint_ms)
    sess._start(timeout)
    return sess


def _websockets_installed():
    import importlib.util
    try:
        return importlib.util.find_spec("websockets") is not None
    except (ImportError, ValueError):
        return False


def _websockets():
    """The parts of python-websockets stt uses, imported on the first connect
    (it costs ~50 ms that batch users and `stt check` never pay)."""
    try:
        from websockets.exceptions import ConnectionClosed, InvalidStatus, InvalidURI, WebSocketException
        from websockets.sync.client import connect as ws_connect
    except ImportError:
        raise StreamError(WEBSOCKETS_MISSING, "config") from None
    import logging
    from types import SimpleNamespace
    # stt reports every failure itself, as a StreamError. The library's own
    # log lines would only repeat it, and one of them is a traceback on stderr
    # each time a keepalive ping finds the network dead.
    logger = logging.getLogger("stt.stream.websockets")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    return SimpleNamespace(connect=ws_connect, ConnectionClosed=ConnectionClosed,
                           InvalidStatus=InvalidStatus, InvalidURI=InvalidURI,
                           WebSocketException=WebSocketException, logger=logger)


def _line(text):
    """Turn text on one line, stripped."""
    return " ".join((text or "").split())


_END = object()  # the last item a session puts on its event queue


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

class Session:
    """One provider session. Use connect(), not the constructor.

    send / poll / finish / close are called from one caller thread. Provider
    frames become events on a queue, from one internal thread (the reader);
    the reader also owns the open turn until the session has ended.

    A provider module subclasses this (or WsSession) and implements _start,
    _send_audio, _send_end and _stop_transport, and calls _partial / _final as
    its frames arrive.
    """

    name = ""
    title = ""                  # for messages
    model_key = ""              # config key naming the realtime model
    default_model = ""
    needs_key = True
    needs_websockets = True
    max_burst_seconds = 0
    max_session_seconds = 0

    def __init__(self, config, sample_rate, terms, languages, speakers, endpoint_ms):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, not {sample_rate}")
        self.config = config
        self.sample_rate = sample_rate
        self.speakers = speakers
        self.endpoint_ms = endpoint_ms
        self.model = self.model_for(config)
        self.notes = []                 # what stt changed in the request
        self.languages = _config.dedupe(languages)
        self.terms = self._cap_terms(_config.dedupe(terms))
        self.timed_out = None           # set by finish() when the provider never confirmed
        self._secrets = _config.api_keys(config)
        self._events = queue.Queue()
        self._done = threading.Event()  # the session ended: _END is on the queue
        self._error = None              # why it ended; None = finished, or ended by us
        self._ending = False            # finish() sent the end of audio
        self._finished = False          # the provider confirmed the end of audio
        self._stopping = False          # we are ending it: errors from here on are ours
        self._over = False              # the caller has been told it ended
        self._send_lock = threading.Lock()
        self._thread = None
        self._sent = 0                  # bytes of audio sent: the session's clock
        self._last_sent = time.monotonic()
        # The open turn (reader thread).
        self._turn = 0
        self._turn_open = False         # it produced a Partial or text
        self._shown = None              # the text of its last Partial
        self._latest = ("", None)       # its text and start, as last heard
        self._letters = {}

    # -- what the provider is, from the config ------------------------------

    @classmethod
    def model_for(cls, config):
        return (config.get(cls.model_key) or "").strip() or cls.default_model

    @classmethod
    def problem(cls, config):
        key = f"{cls.name}_api_key"
        if cls.needs_key and not config.get(key):
            return f"{key} is not set in {_config.path()}"
        if cls.needs_websockets and not _websockets_installed():
            return WEBSOCKETS_MISSING
        return None

    def _cap_terms(self, terms):
        """The key terms this provider takes, keeping the first ones."""
        return terms

    @property
    def info(self):
        return SessionInfo(self.name, self.model, "; ".join(self.notes) or None,
                           self.max_burst_seconds, self.max_session_seconds)

    # -- the caller's side ---------------------------------------------------

    def send(self, pcm):
        """Send audio (raw PCM s16le mono, any length). Raises SessionClosed
        once the session has ended."""
        if self._over or self._done.is_set():
            raise self._closed()
        if not pcm:
            return
        pcm = bytes(pcm)
        try:
            self._send_audio(pcm)
        except Exception as e:
            if not self._transport_error(e):
                raise
            raise self._closed() from None
        self._sent += len(pcm)
        self._last_sent = time.monotonic()

    def poll(self, timeout=0):
        """The events that arrived, waiting up to `timeout` seconds for the
        first one (None: until there is one). Raises SessionClosed once the
        session has ended, carrying the events not returned yet."""
        if self._over:
            raise SessionClosed(self._error, [])
        events = []
        try:
            if timeout is None:
                item = self._events.get()
            elif timeout > 0:
                item = self._events.get(timeout=timeout)
            else:
                item = self._events.get_nowait()
            while item is not _END:
                events.append(item)
                item = self._events.get_nowait()
        except queue.Empty:
            return events
        self._over = True
        self._shutdown()
        raise SessionClosed(self._error, events)

    def finish(self, timeout=10):
        """End of audio: the provider finalizes, the open turn comes back as a
        Final and the session closes. Returns the remaining events.

        Raises SessionClosed(error, events) if the session ends with an error
        meanwhile. When the provider does not confirm within `timeout`, the open
        turn as last heard comes back as the Final, the socket is closed and
        `timed_out` says so. Never takes much longer than `timeout`.
        """
        if self._over or self._done.is_set():
            raise self._closed()
        if not self._sent:
            # No audio, so no turn to finalize. Soniox would even refuse the
            # end of audio ("No audio received.", a 400).
            self._shutdown()
            self._over = True
            events = self._drain()
            if self._error is not None:
                raise SessionClosed(self._error, events)
            return events
        deadline = time.monotonic() + timeout
        # On a stalled network the end of audio (or a keepalive holding the
        # send lock) can block on a full buffer; the watchdog cuts it off.
        watchdog = self._watchdog(timeout)
        try:
            with self._send_lock:
                self._ending = True
                self._send_end()
        except Exception as e:
            if not self._transport_error(e):
                raise
            # The reader reports why the connection went.
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
        events = []
        while (left := deadline - time.monotonic()) > 0:
            try:
                item = self._events.get(timeout=left)
            except queue.Empty:
                break
            if item is _END:
                self._over = True
                self._shutdown()
                if self._error is not None:
                    raise SessionClosed(self._error, events)
                return events
            events.append(item)
        # Timed out: drop the connection, then look at what made it in.
        self._shutdown()
        events += self._drain()
        self._over = True
        if self._finished:
            return events
        if self._error is not None:
            raise SessionClosed(self._error, events)
        self._flush_open_turn()
        events += self._drain()
        self.timed_out = (f"the last turn was not confirmed: {self.title} did not answer "
                          f"the end of audio within {timeout:g} s")
        return events

    def close(self):
        """End the session now: no finals, no waiting. Idempotent; leaves no
        thread behind."""
        self._over = True
        self._shutdown()

    def abort(self):
        """Cut the connection at once, without a goodbye. Unlike the other
        methods it is safe from a signal handler or another thread: a send()
        or finish() stuck on a stalled network fails right away, and the
        caller sees SessionClosed with no error. close() still follows."""
        self._stopping = True
        self._abort_transport()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- shared internals ----------------------------------------------------

    def _closed(self):
        """SessionClosed for a session that has ended, with the events not
        returned yet."""
        if self._over:
            return SessionClosed(self._error, [])
        if not self._done.wait(JOIN_SECONDS) and self._error is None:
            self._error = self._fail(f"connection to {self.title} lost", "network")
        events = self._drain()
        self._over = True
        self._shutdown()
        return SessionClosed(self._error, events)

    def _drain(self):
        events = []
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                return events
            if item is not _END:
                events.append(item)

    def _end(self, error=None):
        """The session is over (reader side): record why, wake the caller."""
        if error is not None and not self._stopping and self._error is None:
            self._error = error
        self._events.put(_END)
        self._done.set()

    def _shutdown(self):
        self._stopping = True
        self._stop_transport()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_SECONDS)

    def _fail(self, message, kind, code=None, retry_after=None):
        """A StreamError whose message is fit for a person: one line, short,
        no API key."""
        return StreamError(rest.clip(rest.scrub(message, self._secrets)), kind, code, retry_after)

    def _transport_error(self, error):
        """Whether `error` from sending means the connection went away."""
        return False

    def _ms(self, nbytes):
        return nbytes * 1000 // (2 * self.sample_rate)

    def _letter(self, speaker):
        if speaker is None or not self.speakers:
            return None
        if speaker not in self._letters:
            self._letters[speaker] = speaker_letter(len(self._letters))
        return self._letters[speaker]

    # -- turns: the same rules for every provider -----------------------------

    def _emit(self, event):
        self._events.put(event)

    def _partial(self, text, start_ms=None):
        """The open turn now reads `text`. A Partial goes out when it changed;
        an empty text does not open a turn."""
        text = _line(text)
        if not text and not self._turn_open:
            return
        self._turn_open = True
        self._latest = (text, start_ms)
        if text != self._shown:
            self._shown = text
            self._emit(Partial(self._turn, text, start_ms))

    def _final(self, text, start_ms=None, end_ms=None, speaker=None, language=None):
        """The open turn ended with `text`. A turn with neither text nor
        Partials leaves no trace; one that showed Partials always gets its
        Final, even an empty one, so callers can clear the tail."""
        text = _line(text)
        if text or self._turn_open:
            self._emit(Final(self._turn, text, start_ms, end_ms, speaker, language))
            self._turn += 1
        self._turn_open, self._shown, self._latest = False, None, ("", None)

    def _flush_open_turn(self):
        """Close the open turn as last heard (best effort: unconfirmed)."""
        if self._turn_open:
            self._final(*self._latest)

    # -- provider hooks --------------------------------------------------------

    def _start(self, timeout):
        raise NotImplementedError

    def _send_audio(self, pcm):
        raise NotImplementedError

    def _send_end(self):
        raise NotImplementedError

    def _stop_transport(self):
        pass

    def _abort_transport(self):
        """Drop the connection, touching no lock (see abort)."""

    def _watchdog(self, seconds):
        """A started timer that aborts the session after `seconds`, for a
        step the network could block; None when nothing can block."""
        return None


class WsSession(Session):
    """A provider behind one websocket (python-websockets' sync client).

    Its keepalive pings (every 20 s, 20 s timeout: the library defaults) turn a
    silently dead network into a close, which the reader reports as a network
    error; while audio flows, TCP_USER_TIMEOUT does the same for audio the
    network stopped taking. The reader thread also sends the provider's own
    keepalive frame while no audio flows.
    """

    url = ""
    keepalive_frame = None      # the provider's keepalive message, if it has one
    rate_limit_retry = None     # seconds, for a 429 on the upgrade

    def __init__(self, *args):
        super().__init__(*args)
        self._ws = None
        self._lib = None

    # -- provider hooks --------------------------------------------------------

    def _connect_url(self):
        return self.url

    def _headers(self):
        raise NotImplementedError

    def _accept(self, deadline):
        """Wait until the provider accepted the session; StreamError if not."""
        raise NotImplementedError

    def _handle(self, message):
        """Turn one provider frame (parsed JSON) into events. Returns True when
        the provider confirmed the end of audio; raises StreamError for an error
        frame that ends the session."""
        raise NotImplementedError

    def _closed_error(self, exc):
        """StreamError for a connection that closed under us (ConnectionClosed)."""
        rcvd = exc.rcvd
        reason = f" ({rcvd.reason})" if rcvd and rcvd.reason else ""
        return self._fail(f"connection to {self.title} lost{reason}", "network",
                          code=rcvd.code if rcvd else 1006)

    # -- connection ------------------------------------------------------------

    def _start(self, timeout):
        deadline = time.monotonic() + timeout
        self._lib = lib = _websockets()
        try:
            self._ws = lib.connect(
                self._connect_url(), additional_headers=self._headers(), open_timeout=timeout,
                ping_interval=PING_SECONDS, ping_timeout=PING_SECONDS, close_timeout=CLOSE_SECONDS,
                logger=lib.logger,
            )
        except Exception as e:
            raise self._connect_error(e) from None
        if hasattr(socket, "TCP_USER_TIMEOUT"):
            try:
                self._ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT,
                                           int(STALL_SECONDS * 1000))
            except OSError:  # not TCP after all: the pings are still there
                pass
        try:
            self._accept(deadline)
        except BaseException as e:
            self._stopping = True
            self._stop_transport()
            if isinstance(e, lib.ConnectionClosed):
                raise self._closed_error(e) from None
            if isinstance(e, (OSError, lib.WebSocketException)):
                raise self._fail(f"connection to {self.title} failed: {e}", "network") from None
            if isinstance(e, ValueError):
                raise self._fail(f"unexpected answer from {self.title}: {e}", "internal") from None
            raise
        self._thread = threading.Thread(target=self._read, name=f"stt-stream-{self.name}", daemon=True)
        self._thread.start()

    def _connect_error(self, e):
        lib = self._lib
        if isinstance(e, lib.InvalidStatus):
            status = e.response.status_code
            if status in (401, 403):
                kind = "auth"
            elif status == 429:
                kind = "rate_limit"
            elif status >= 500:
                kind = "server"
            else:
                kind = "rejected"
            return self._fail(f"{self.title} refused the connection: HTTP {status}", kind, code=status,
                              retry_after=self.rate_limit_retry if kind == "rate_limit" else None)
        if isinstance(e, lib.InvalidURI):
            return self._fail(f"bad {self.title} URL: {e}", "internal")
        if isinstance(e, (OSError, lib.WebSocketException)):
            return self._fail(f"cannot reach {self.title}: {e or type(e).__name__}", "network")
        return self._fail(f"internal error connecting to {self.title}: {e!r}", "internal")

    def _transport_error(self, error):
        lib = self._lib
        return isinstance(error, (OSError, lib.WebSocketException)) if lib else False

    def _stop_transport(self):
        ws = self._ws
        if ws is None:
            return
        try:
            ws.close()   # idempotent; waits at most CLOSE_SECONDS for the provider
        except Exception:  # noqa: BLE001 - the socket is gone either way
            pass
        me = threading.current_thread()
        for name in ("recv_events_thread", "keepalive_thread"):
            thread = getattr(ws, name, None)
            if thread is not None and thread is not me:
                thread.join(JOIN_SECONDS)

    def _abort_transport(self):
        ws = self._ws
        if ws is None:
            return
        try:
            # The plain socket's shutdown: an SSL socket's own also drops its
            # SSL state under whichever thread is using it.
            socket.socket.shutdown(ws.socket, socket.SHUT_RDWR)
        except OSError:  # closed already
            pass

    def _watchdog(self, seconds):
        timer = threading.Timer(seconds, self.abort)
        timer.daemon = True
        timer.start()
        return timer

    # -- the reader thread -------------------------------------------------------

    def _read(self):
        error = None
        try:
            while True:
                try:
                    message = self._ws.recv(timeout=self._tick())
                except TimeoutError:
                    self._keepalive()
                    continue
                if self._handle(json.loads(message)):
                    if not self._ending:
                        raise self._fail(f"{self.title} ended the session unasked", "server")
                    self._finished = True
                    break
        except self._lib.ConnectionClosed as e:
            error = self._closed_error(e)
        except StreamError as e:
            error = e
        except OSError as e:
            error = self._fail(f"connection to {self.title} failed: {e}", "network")
        except Exception as e:  # noqa: BLE001 - a bug or a frame stt does not understand
            error = self._fail(f"internal error reading from {self.title}: {e!r}", "internal")
        finally:
            # The connection goes before the caller hears of the end, so the
            # provider stops billing as soon as possible.
            self._stop_transport()
            try:
                if error is not None and not self._stopping:
                    # Ended under the caller: the open turn as last heard is
                    # its Final, so every turn that showed a Partial gets one.
                    self._flush_open_turn()
            finally:
                self._end(error)

    def _tick(self):
        """How long the reader may wait for a frame."""
        if self.keepalive_frame is None or self._ending:
            return 1.0
        due = self._last_sent + KEEPALIVE_SECONDS - time.monotonic()
        return min(1.0, max(0.01, due))

    def _keepalive(self):
        if self.keepalive_frame is None:
            return
        with self._send_lock:
            if not self._ending and time.monotonic() - self._last_sent >= KEEPALIVE_SECONDS:
                self._ws.send(self.keepalive_frame)
                self._last_sent = time.monotonic()
