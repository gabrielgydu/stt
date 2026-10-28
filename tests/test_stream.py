#!/usr/bin/env python3
"""
Exercise stt.stream and `stt stream` against local mocks of the Soniox and
AssemblyAI real-time APIs: no network, no credit, and never the real config.

    python3 tests/test_stream.py

MockSoniox speaks the protocol as Soniox documents it: a start message, binary
audio, an empty text frame to end, a `finished` response, and error frames
right before a 1000 close. MockAAI is audiorecorder's AssemblyAI mock
(audiorecorder/tests/test_stream.py), extended. Both listen on 127.0.0.1; the
providers' class attribute `url` points at them, and `stt stream` child
processes get it through a small -c wrapper, as test_stt.py does for HTTP.
"""

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stt import stream  # noqa: E402
from stt.stream import Final, Partial, SessionClosed, StreamError  # noqa: E402
from stt.stream import soniox as soniox_module  # noqa: E402
from stt.stream.assemblyai import AssemblyAI  # noqa: E402
from stt.stream.soniox import Soniox  # noqa: E402

failures = []
checks = 0

BPS = 32000          # bytes per second: 16 kHz s16le mono
GRACE = 0.1          # Soniox's accept grace, shortened for the tests
KEY_S = "sk-test-secret-soniox"
KEY_A = "aai-test-secret"
CFG_S = {"provider": "soniox", "soniox_api_key": KEY_S, "languages": "pt,en", "keyterms": "Kubernetes, PostgreSQL"}
CFG_A = {"provider": "assemblyai", "assemblyai_api_key": KEY_A, "keyterms": "Kubernetes"}


def check(label, condition, detail=""):
    global checks
    checks += 1
    if condition:
        print(f"  ok    {label}")
    else:
        failures.append(label)
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def section(name):
    print(f"\n{name}")


# --------------------------------------------------------------------------
# mock Soniox
# --------------------------------------------------------------------------

def tok(text, final, start=None, end=None, speaker=None, language=None):
    token = {"text": text, "is_final": final, "confidence": 0.9}
    if start is not None:
        token["start_ms"], token["end_ms"] = start, end if end is not None else start + 100
    if speaker is not None:
        token["speaker"] = speaker
    if language is not None:
        token["language"] = language
    return token


def resp(*tokens, **extra):
    return {"tokens": list(tokens), "final_audio_proc_ms": 0, "total_audio_proc_ms": 0, **extra}


def soniox_error(code, error_type, message):
    return {"tokens": [], "error_code": code, "error_type": error_type, "error_message": message,
            "request_id": "req-1"}


class MockSoniox:
    """One scenario (a dict) per connection, in order; {} once they run out.

    script        responses sent one per audio frame received
    auto          True: one word per 250 ms of audio, a turn (<end>) per 4 words
    on_end        responses sent after "" (then `finished`); "silent" = never
                  answer; a dict = an error frame (then close 1000). With no
                  audio received, "" gets Soniox's documented 400 "No audio
                  received." whatever the scenario says
    reject        an error frame sent right after the start message
    drop_start    drop the TCP connection right after the start message
    error_after   (n, frame): after n audio frames, the error frame + close 1000
    drop_after    n: after n audio frames, drop the TCP connection (no close frame)
    http          refuse the upgrade with this HTTP status
    """

    def __init__(self):
        self.scenarios, self.sessions = [], []
        self.lock = threading.Lock()
        self.server = serve(self.handle, "127.0.0.1", 0, process_request=self.check)
        self.port = self.server.socket.getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}/transcribe-websocket"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reset(self, scenarios=()):
        self.scenarios, self.sessions = list(scenarios), []

    def stop(self):
        self.server.shutdown()

    def check(self, connection, request):
        with self.lock:
            index = len(self.sessions)
            scenario = self.scenarios[index] if index < len(self.scenarios) else {}
            rec = {"auth": request.headers.get("Authorization"), "path": request.path, "start": None,
                   "bytes": 0, "frames": 0, "texts": [], "done": threading.Event()}
            self.sessions.append(rec)
        connection.mock = (rec, scenario)
        if "http" in scenario:
            rec["done"].set()
            return connection.respond(scenario["http"], "refused\n")
        return None

    def handle(self, ws):
        rec, scenario = ws.mock
        try:
            self.serve_session(ws, rec, scenario)
        except ConnectionClosed:
            pass
        finally:
            rec["done"].set()

    @staticmethod
    def drop(ws):
        ws.socket.shutdown(socket.SHUT_RDWR)

    def serve_session(self, ws, rec, sc):
        rec["start"] = json.loads(ws.recv(timeout=5))
        if "reject" in sc:
            ws.send(json.dumps(sc["reject"]))
            ws.close(1000)
            return
        if sc.get("drop_start"):
            return self.drop(ws)
        script = list(sc.get("script", []))
        turn_from = 0   # auto: first word of the open turn
        for msg in ws:
            if isinstance(msg, bytes):
                rec["bytes"] += len(msg)
                rec["frames"] += 1
                if script:
                    ws.send(json.dumps(script.pop(0)))
                if sc.get("auto"):
                    words = rec["bytes"] // (BPS // 4)
                    tokens = []
                    if words - turn_from >= 4:
                        tokens += [tok(f" w{i}", True, i * 250, i * 250 + 250, language="en")
                                   for i in range(turn_from, turn_from + 4)] + [tok("<end>", True)]
                        turn_from += 4
                    tokens += [tok(f" w{i}", False, i * 250, i * 250 + 250, language="en")
                               for i in range(turn_from, words)]
                    ws.send(json.dumps(resp(*tokens)))
                if "error_after" in sc and rec["frames"] >= sc["error_after"][0]:
                    ws.send(json.dumps(sc["error_after"][1]))
                    ws.close(1000)
                    return
                if "drop_after" in sc and rec["frames"] >= sc["drop_after"]:
                    return self.drop(ws)
                continue
            rec["texts"].append(msg)
            if msg != "":
                continue
            if not rec["bytes"]:
                ws.send(json.dumps(soniox_error(400, "invalid_request", "No audio received.")))
                ws.close(1000)
                return
            on_end = sc.get("on_end", [])
            if on_end == "silent":
                continue
            if isinstance(on_end, dict):
                ws.send(json.dumps(on_end))
                ws.close(1000)
                return
            for r in on_end:
                ws.send(json.dumps(r))
            if sc.get("auto"):
                words = rec["bytes"] // (BPS // 4)
                ws.send(json.dumps(resp(*[tok(f" w{i}", True, i * 250, i * 250 + 250, language="en")
                                          for i in range(turn_from, words)])))
            ws.send(json.dumps(resp(finished=True)))
            ws.close(1000)
            return

    def keepalives(self, rec):
        return sum(1 for t in rec["texts"] if t and json.loads(t).get("type") == "keepalive")


# --------------------------------------------------------------------------
# mock AssemblyAI (from audiorecorder's tests/test_stream.py, extended)
# --------------------------------------------------------------------------

class MockAAI:
    """One behaviour per connection, in order; "normal" once they run out.

    normal          transcribe until Terminate (one finished turn per second of audio)
    double          like normal, but every finished turn also comes unformatted
                    first (Universal-Streaming with format_turns)
    close:CODE:S    close with CODE after S seconds of audio, no error frame
    error:CODE:S:T  error frame (text T) + close with CODE after S seconds of audio
    expire:CODE:S   an unasked Termination, then close with CODE, after S seconds
    errterm:CODE:S:T  error frame, an unasked Termination, close with CODE
    drop:S          drop the TCP connection after S seconds of audio (no close frame)
    reject:CODE:T   error frame + close with CODE before Begin
    nobegin         never send Begin
    silent          partials only: no finished turn, and Terminate is never answered
    """

    def __init__(self):
        self.behaviours, self.sessions = [], []
        self.lock = threading.Lock()
        self.server = serve(self.handle, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}/v3/ws"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reset(self, behaviours=()):
        self.behaviours, self.sessions = list(behaviours), []

    def stop(self):
        self.server.shutdown()

    def handle(self, ws):
        with self.lock:
            index = len(self.sessions)
            rec = {
                "query": urllib.parse.parse_qs(urllib.parse.urlparse(ws.request.path).query),
                "auth": ws.request.headers.get("Authorization"),
                "bytes": 0, "chunks": [], "audio": bytearray(), "texts": [], "done": threading.Event(),
            }
            self.sessions.append(rec)
        behaviour = self.behaviours[index] if index < len(self.behaviours) else "normal"
        try:
            self.serve_session(ws, rec, index, behaviour)
        except ConnectionClosed:
            pass
        finally:
            rec["done"].set()

    def serve_session(self, ws, rec, index, behaviour):
        kind, _, arg = behaviour.partition(":")
        if kind == "reject":
            code, _, text = arg.partition(":")
            ws.send(json.dumps({"type": "Error", "error_code": int(code), "error": text}))
            ws.close(int(code), "See Error message for details")
            return
        if kind == "nobegin":
            for _ in ws:
                pass
            return

        ws.send(json.dumps({"type": "Begin", "id": f"mock-{index}", "expires_at": 0}))
        turn, turn_from = 0, 0
        for msg in ws:
            if isinstance(msg, bytes):
                rec["bytes"] += len(msg)
                rec["chunks"].append(len(msg))
                rec["audio"] += msg
                if rec["bytes"] - turn_from >= BPS:
                    text = f"s{index} t{turn}"
                    words = [{"text": text, "start": turn_from * 1000 // BPS, "end": rec["bytes"] * 1000 // BPS}]
                    ws.send(json.dumps({"type": "Turn", "turn_order": turn, "end_of_turn": False,
                                        "turn_is_formatted": False, "transcript": text[:4], "words": words}))
                    if kind == "double":
                        ws.send(json.dumps({"type": "Turn", "turn_order": turn, "end_of_turn": True,
                                            "turn_is_formatted": False, "transcript": text, "words": words}))
                        text = text.capitalize() + "."
                    if kind != "silent":
                        ws.send(json.dumps({"type": "Turn", "turn_order": turn, "end_of_turn": True,
                                            "turn_is_formatted": True, "transcript": text, "words": words}))
                        turn, turn_from = turn + 1, rec["bytes"]
                if kind in ("close", "error", "expire", "errterm", "drop"):
                    parts = arg.split(":", 2)
                    seconds = float(parts[0] if kind == "drop" else parts[1])
                    if rec["bytes"] >= seconds * BPS:
                        if kind == "drop":
                            ws.socket.shutdown(socket.SHUT_RDWR)
                            return
                        if kind in ("error", "errterm"):
                            ws.send(json.dumps({"type": "Error", "error_code": int(parts[0]), "error": parts[2]}))
                        if kind in ("expire", "errterm"):
                            ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": 10800}))
                        ws.close(int(parts[0]), "Session closed by the mock" if kind in ("close", "expire")
                                 else "See Error message for details")
                        return
            else:
                rec["texts"].append(msg)
                if json.loads(msg).get("type") != "Terminate" or kind == "silent":
                    continue
                if rec["bytes"] > turn_from:
                    words = [{"text": "end", "start": turn_from * 1000 // BPS, "end": rec["bytes"] * 1000 // BPS}]
                    ws.send(json.dumps({"type": "Turn", "turn_order": turn, "end_of_turn": True,
                                        "turn_is_formatted": True, "transcript": f"s{index} end",
                                        "words": words}))
                ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": rec["bytes"] // BPS}))
                ws.close(1000, "Session Ended")
                return


# --------------------------------------------------------------------------
# a stalled network
# --------------------------------------------------------------------------

class StalledPeer:
    """A websocket server that completes the opening handshake and then never
    reads, sends or answers a ping again: a network that stalled. Its small
    receive buffer makes a sender's buffer fill within a few MB."""

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.url = f"ws://127.0.0.1:{self.listener.getsockname()[1]}/x"
        self.conns, self.stopped = [], threading.Event()
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            self.conns.append(conn)
            threading.Thread(target=self.handshake, args=(conn,), daemon=True).start()

    def handshake(self, conn):
        import base64
        import hashlib
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                return
            data += chunk
        key = ""
        for line in data.split(b"\r\n\r\n", 1)[0].decode().split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()
        accept = base64.b64encode(hashlib.sha1((key + self.GUID).encode()).digest()).decode()
        conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        self.stopped.wait()

    def stop(self):
        self.stopped.set()
        self.listener.close()
        for conn in self.conns:
            conn.close()


def bounded(fn, seconds, sess):
    """fn() on a helper thread, given `seconds`. Returns (result or exception,
    seconds taken), or ("blocked", seconds) after cutting the session's socket
    from outside, so a failing check never hangs the run."""
    box = {}

    def call():
        try:
            box["out"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the test
            box["out"] = e

    t0 = time.monotonic()
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        try:
            socket.socket.shutdown(sess._ws.socket, socket.SHUT_RDWR)
        except OSError:
            pass
        thread.join(5)
        return "blocked", time.monotonic() - t0
    return box.get("out"), time.monotonic() - t0


def send_forever(sess):
    while True:
        sess.send(silence(2))


def fill_send_buffer(sess):
    """Fill the session socket's send buffer without blocking (raw bytes: the
    stalled peer never reads them anyway)."""
    while True:
        try:
            sess._ws.socket.send(b"\0" * 65536, socket.MSG_DONTWAIT)
        except BlockingIOError:
            return


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def collect(sess, seconds=3.0, until=None):
    """Poll for up to `seconds` (or until `until(events)`). Returns
    (events, SessionClosed or None)."""
    events = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            events += sess.poll(0.05)
        except SessionClosed as e:
            return events + e.events, e
        if until and until(events):
            break
    return events, None


def silence(seconds, byte=b"\0"):
    return byte * int(seconds * BPS)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def settle_threads(seconds=1.0):
    """threading.active_count() once it stops changing."""
    count, deadline = threading.active_count(), time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.05)
        now = threading.active_count()
        if now == count:
            return now
        count = now
    return count


def threads_back_to(baseline, seconds=2.0):
    deadline = time.monotonic() + seconds
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.02)
    return threading.active_count()


def error_of(fn):
    try:
        fn()
    except StreamError as e:
        return e
    return None


def write_config(path, **values):
    with open(path, "w") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


# --------------------------------------------------------------------------
# tests: library
# --------------------------------------------------------------------------

def test_imports():
    section("imports")

    def modules(code):
        r = subprocess.run([sys.executable, "-c", f"import sys, json; sys.path.insert(0, {ROOT!r}); {code}"],
                           capture_output=True, text=True)
        return r.stdout.strip(), r.stderr.strip()

    out, err = modules("import stt; print(json.dumps(['websockets' in sys.modules, 'stt.stream' in sys.modules]))")
    check("import stt loads neither websockets nor stt.stream", out == "[false, false]", (out, err))
    out, err = modules("import stt.stream, stt.stream_cli, stt.cli; print('websockets' in sys.modules)")
    check("import stt.stream / stt.stream_cli / stt.cli does not load websockets", out == "False", (out, err))
    out, err = modules(
        "sys.modules['websockets'] = None; from stt import stream;"
        "p = stream.config_problem({'soniox_api_key': 'k'});"
        "e = None\ntry: stream.connect(config={'soniox_api_key': 'k'})\n"
        "except stream.StreamError as x: e = x\n"
        "s = stream.connect(config={'provider': 'fake'}); s.close();"
        "print(json.dumps([p, e.kind, e.fatal]))"
    )
    data = json.loads(out) if out.startswith("[") else [None, None, None]
    check("without websockets: stt.stream imports and config_problem says so",
          data[0] and "python-websockets" in data[0], (out, err))
    check("without websockets: connect raises StreamError(config)", data[1:] == ["config", True], (out, err))


def test_config_api(ms):
    section("config_problem / model_name")
    path = os.environ["STT_CONFIG"]
    check("unknown provider", "unknown provider 'whisper'" in (stream.config_problem({"provider": "whisper"}) or ""))
    check("soniox needs its key (named, with the config path)",
          stream.config_problem({}) == f"soniox_api_key is not set in {path}", stream.config_problem({}))
    check("assemblyai needs its own key",
          stream.config_problem({"provider": "assemblyai", "soniox_api_key": "k"})
          == f"assemblyai_api_key is not set in {path}")
    check("provider_name overrides the config",
          stream.config_problem(CFG_S, provider_name="assemblyai") == f"assemblyai_api_key is not set in {path}")
    check("key set -> no problem", stream.config_problem(CFG_S) is None and stream.config_problem(CFG_A) is None)
    check("fake needs no key", stream.config_problem({"provider": "fake"}) is None)
    check("bad fake_fail is a config problem",
          "fake_fail" in (stream.config_problem({"provider": "fake", "fake_fail": "soon"}) or ""))
    check("file-transcription knobs (submit:, job, slow:) are ignored",
          [stream.config_problem({"provider": "fake", "fake_fail": v}) for v in ("submit:auth", "job", "slow:5")]
          == [None, None, None])
    check("default models",
          (stream.model_name(CFG_S), stream.model_name(CFG_A), stream.model_name({"provider": "fake"}))
          == ("stt-rt-v5", "universal-3-6-pro", "fake"))
    check("soniox_stream_model / assemblyai_stream_model override",
          stream.model_name({**CFG_S, "soniox_stream_model": "stt-rt-v4"}) == "stt-rt-v4"
          and stream.model_name({**CFG_A, "assemblyai_stream_model": "universal-streaming-english"})
          == "universal-streaming-english")
    e = error_of(lambda: stream.model_name({"provider": "whisper"}))
    check("model_name of an unknown provider: StreamError(config)", e is not None and e.kind == "config")

    ms.reset()
    e = error_of(lambda: stream.connect(config={"provider": "soniox"}))
    check("connect without a key: StreamError(config, fatal), before any network",
          e is not None and e.kind == "config" and e.fatal and ms.sessions == [], (e, ms.sessions))

    unreadable = os.path.join(os.path.dirname(path), "config-dir")
    os.makedirs(unreadable, exist_ok=True)
    os.environ["STT_CONFIG"] = unreadable
    try:
        problem = stream.config_problem()
        e = error_of(lambda: stream.connect())
        e_model = error_of(lambda: stream.model_name())
    except OSError as raw:
        problem = e = e_model = raw
    finally:
        os.environ["STT_CONFIG"] = path
    check("unreadable config: config_problem says so (no OSError)",
          isinstance(problem, str) and problem.startswith(f"cannot read {unreadable}"), repr(problem))
    check("unreadable config: connect and model_name raise StreamError(config)",
          isinstance(e, StreamError) and e.kind == "config" and isinstance(e_model, StreamError)
          and e_model.kind == "config", (repr(e), repr(e_model)))


def test_soniox_handshake(ms):
    section("soniox: handshake and start message")
    ms.reset([{}])
    sess = stream.connect(config=CFG_S, speakers=True, endpoint_ms=1500)
    rec = ms.sessions[0]
    check("bearer key in the Authorization header", rec["auth"] == f"Bearer {KEY_S}", rec["auth"])
    check("start message fields", rec["start"] == {
        "model": "stt-rt-v5", "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
        "enable_language_identification": True, "enable_speaker_diarization": True,
        "enable_endpoint_detection": True, "language_hints": ["pt", "en"], "max_endpoint_delay_ms": 1500,
        "context": {"terms": ["Kubernetes", "PostgreSQL"]},
    }, rec["start"])
    check("no api_key in the start message", "api_key" not in json.dumps(rec["start"]))
    info = sess.info
    check("info: provider, model, limits, no note",
          (info.provider, info.model, info.note, info.max_burst_seconds, info.max_session_seconds)
          == ("soniox", "stt-rt-v5", None, 10, 18000), info)
    sess.close()

    ms.reset([{}])
    sess = stream.connect(config=CFG_S, terms=[], languages=[], endpoint_ms=100, sample_rate=8000)
    start = ms.sessions[0]["start"]
    check("overrides: no hints, no context, speakers off, rate passed",
          "language_hints" not in start and "context" not in start
          and start["enable_speaker_diarization"] is False and start["sample_rate"] == 8000, start)
    check("endpoint_ms clamped to 500-3000, with a note",
          start["max_endpoint_delay_ms"] == 500 and "500" in (sess.info.note or ""), (start, sess.info.note))
    sess.close()


SONIOX_TURNS = [
    resp(tok("Bom", False, 0, 200, language="pt")),
    resp(tok("Bom", False, 0, 200, language="pt"), tok(" dia", False, 200, 400, language="pt")),
    resp(tok("Bom", False, 0, 200, language="pt"), tok(" dia", False, 200, 400, language="pt")),
    resp(tok("Bom", True, 0, 200, language="pt"), tok(" dia", True, 200, 400, language="pt"),
         tok(",", False, 400, 450, language="pt")),
    resp(tok(",", True, 400, 450, language="pt"), tok(" Ana.", True, 450, 900, language="pt"),
         tok("<end>", True), tok(" Hi", False, 1000, 1200, language="en")),
    resp(),
    resp(tok("<end>", True)),
    resp(tok("<end>", True)),
    resp(tok(" Tudo", True, 2000, 2200, language="pt"), tok(" bem?", True, 2200, 2500, language="pt"),
         tok("<fin>", True)),
    resp(tok(" Até", False, 3000, 3200, language="pt")),
]
SONIOX_TURNS_END = [resp(tok(" Até", True, 3000, 3200, language="pt"), tok(" logo.", True, 3200, 3500, language="pt"))]

SONIOX_SPEAKERS = [
    resp(tok("Oi", True, 0, 200, "1", "pt"), tok(" tudo", True, 200, 400, "1", "pt"),
         tok(" bem", False, 400, 600, "1", "pt")),
    resp(tok(" bem?", True, 400, 600, "1", "pt"), tok(" Hi", True, 1000, 1200, "2", "en"),
         tok(" there.", True, 1200, 1500, "2", "en")),
    resp(tok("<end>", True)),
    resp(tok(" Tchau.", True, 2000, 2200, "1", "pt"), tok("<end>", True)),
]


def run_script(ms, script, on_end=(), speakers=False, frames=None):
    ms.reset([{"script": script, "on_end": list(on_end)}])
    sess = stream.connect(config=CFG_S, speakers=speakers)
    for _ in range(frames or len(script)):
        sess.send(silence(0.1))
    return sess, sess.finish(3)


def test_soniox_turns(ms):
    section("soniox: tokens to turns")
    sess, events = run_script(ms, SONIOX_TURNS, SONIOX_TURNS_END)
    expected = [
        Partial(0, "Bom", 0),
        Partial(0, "Bom dia", 0),
        Partial(0, "Bom dia,", 0),
        Final(0, "Bom dia, Ana.", 0, 900, None, "pt"),
        Partial(1, "Hi", 1000),
        Partial(1, ""),
        Final(1, ""),
        Final(2, "Tudo bem?", 2000, 2500, None, "pt"),
        Partial(3, "Até", 3000),
        Partial(3, "Até logo.", 3000),
        Final(3, "Até logo.", 3000, 3500, None, "pt"),
    ]
    check("events match turn by turn (finals across responses, tail replaced, <end>/<fin>)",
          events == expected, "\n        ".join(map(str, events)))
    partials = [(e.turn, e.text) for e in events if isinstance(e, Partial)]
    check("a Partial only when the text changed", len(partials) == len(set(partials)), partials)
    finals = [e.turn for e in events if isinstance(e, Final)]
    check("exactly one Final per turn, turns numbered in order", finals == [0, 1, 2, 3], finals)
    check("a turn whose partial vanished still ends with an (empty) Final",
          Final(1, "") in events and not any(e.turn == 1 for e in events[events.index(Final(1, "")) + 1:]))
    check("an <end> with neither text nor partials makes nothing", len(finals) == 4)

    sess, events = run_script(ms, SONIOX_SPEAKERS, speakers=True)
    finals = [e for e in events if isinstance(e, Final)]
    check("speakers on: a speaker change among finals closes the turn",
          [(f.turn, f.text, f.speaker, f.language) for f in finals]
          == [(0, "Oi tudo bem?", "A", "pt"), (1, "Hi there.", "B", "en"), (2, "Tchau.", "A", "pt")],
          "\n        ".join(map(str, events)))
    check("start_ms / end_ms span the turn's tokens",
          [(f.start_ms, f.end_ms) for f in finals] == [(0, 600), (1000, 1500), (2000, 2200)])
    check("Partial before the split carried the tail", events[0] == Partial(0, "Oi tudo bem", 0), events[:1])

    sess, events = run_script(ms, SONIOX_SPEAKERS, speakers=False)
    finals = [(e.text, e.speaker) for e in events if isinstance(e, Final)]
    check("speakers off: no split, no speaker labels",
          finals == [("Oi tudo bem? Hi there.", None), ("Tchau.", None)], finals)


def test_soniox_finish(ms):
    section("soniox: finish")
    script = [resp(tok("Olá", False, 0, 300, language="pt"))]
    ms.reset([{"script": script, "on_end": [resp(tok("Olá", True, 0, 300, language="pt"),
                                                tok(" mundo.", True, 300, 700, language="pt"))]}])
    sess = stream.connect(config=CFG_S)
    sess.send(silence(0.1))
    events, _ = collect(sess, 1.0, until=lambda ev: ev)
    t0 = time.monotonic()
    rest = sess.finish(3)
    rec = ms.sessions[0]
    check("finish sends one empty text frame", rec["texts"] == [""], rec["texts"])
    check("the open turn comes back as a Final, after `finished`",
          events == [Partial(0, "Olá", 0)]
          and rest == [Partial(0, "Olá mundo.", 0), Final(0, "Olá mundo.", 0, 700, None, "pt")], (events, rest))
    check("finish returns promptly", time.monotonic() - t0 < 1, time.monotonic() - t0)
    closed = None
    try:
        sess.poll()
    except SessionClosed as e:
        closed = e
    check("afterwards: SessionClosed(error=None)", closed is not None and closed.error is None, closed)
    try:
        sess.send(b"\0\0")
        check("send after finish raises SessionClosed", False)
    except SessionClosed as e:
        check("send after finish raises SessionClosed", e.error is None)
    check("the provider saw the socket closed", rec["done"].wait(2))

    ms.reset([{"script": [resp(tok("Olá", True, 0, 300, language="pt"), tok(" mun", False, 300, 500))],
               "on_end": "silent"}])
    sess = stream.connect(config=CFG_S)
    sess.send(silence(0.1))
    collect(sess, 1.0, until=lambda ev: ev)
    t0 = time.monotonic()
    rest = sess.finish(0.4)
    took = time.monotonic() - t0
    check("finish timeout: the open turn as last heard comes back as the Final",
          rest == [Final(0, "Olá mun", 0, 500, None, "pt")], rest)
    check("finish timeout: timed_out says so", "not confirmed" in (sess.timed_out or ""), sess.timed_out)
    check("finish timeout: no hang past timeout + 1 s", took < 1.4, took)
    check("finish timeout: the socket is closed", ms.sessions[0]["done"].wait(2))

    ms.reset([{}])
    sess = stream.connect(config=CFG_S)
    try:
        rest = sess.finish(3)
    except SessionClosed as e:
        rest = e
    check("finish with no audio: a clean end, not Soniox's 400 'No audio received.'",
          rest == [] and sess.timed_out is None, repr(rest))
    check("finish with no audio: no end-of-audio frame sent, the socket closed",
          ms.sessions[0]["done"].wait(2) and ms.sessions[0]["texts"] == [], ms.sessions[0]["texts"])

    ms.reset([{"script": [resp(tok("Olá", False, 0, 300))],
               "on_end": soniox_error(503, "service_unavailable", "Cannot continue request (code 11).")}])
    sess = stream.connect(config=CFG_S)
    sess.send(silence(0.1))
    collect(sess, 1.0, until=lambda ev: ev)
    try:
        sess.finish(3)
        check("an error during finish raises SessionClosed", False)
    except SessionClosed as e:
        check("an error during finish raises SessionClosed(error)",
              e.error is not None and e.error.kind == "server" and "Cannot continue" in str(e.error), e.error)


SONIOX_ERROR_ROWS = [
    # (code, error_type, kind, fatal, routine, retry_after)
    (400, "invalid_request", "rejected", True, False, None),
    (401, "unauthenticated", "auth", True, False, None),
    (402, "organization_balance_exhausted", "credit", True, False, None),
    (403, "permission_denied", "auth", True, False, None),
    (403, "temp_api_key_session_expired", "limit", False, True, None),
    (408, "request_timeout", "network", False, False, None),
    (413, "max_duration_reached", "limit", False, True, None),
    (429, "limit_exceeded", "rate_limit", False, False, 20),
    (500, "internal_error", "server", False, False, None),
    (503, "service_unavailable", "server", False, False, None),
]


def test_soniox_errors(ms):
    section("soniox: error frames mid-session")
    for code, etype, kind, fatal, routine, retry in SONIOX_ERROR_ROWS:
        message = f"mock {etype} for key {KEY_S}"
        ms.reset([{"script": [resp(tok("Oi", False, 0, 200))],
                   "error_after": (2, soniox_error(code, etype, message))}])
        sess = stream.connect(config=CFG_S)
        sess.send(silence(0.1))
        sess.send(silence(0.1))
        events, closed = collect(sess, 3)
        e = closed.error if closed else None
        check(f"{code} {etype} -> {kind}, fatal={fatal}, routine={routine}, retry_after={retry}",
              e is not None and (e.kind, e.fatal, e.routine, e.retry_after, e.code) == (kind, fatal, routine, retry, code),
              e and (e.kind, e.fatal, e.routine, e.retry_after, e.code))
        if code == 401:
            check("the message is the provider's, with the key scrubbed",
                  str(e) == "mock unauthenticated for key ***", str(e))
            check("events before the error ride on SessionClosed, the open turn closed as last heard",
                  events == [Partial(0, "Oi", 0), Final(0, "Oi", 0, 200)], events)
            try:
                sess.send(b"\0\0")
                check("send after the error raises SessionClosed again", False)
            except SessionClosed as again:
                check("send after the error raises SessionClosed again (no events twice)",
                      again.error is e and again.events == [])
        sess.close()

    ms.reset([{"script": [resp(tok("Olá", True, 0, 300, language="pt"), tok(" mun", False, 300, 500))],
               "drop_after": 1}])
    sess = stream.connect(config=CFG_S)
    sess.send(silence(0.1))
    events, closed = collect(sess, 3)
    e = closed.error if closed else None
    check("connection dropped without an error frame -> network, transient",
          e is not None and e.kind == "network" and not e.fatal and not e.routine, e and e.kind)
    check("dropped: every turn that showed a Partial ends with a Final (as last heard)",
          events == [Partial(0, "Olá mun", 0), Final(0, "Olá mun", 0, 500, None, "pt")], events)


def test_connect_failures(ms, ma):
    section("connect failures")
    port = free_port()
    old = Soniox.url
    Soniox.url = f"ws://127.0.0.1:{port}/x"
    e = error_of(lambda: stream.connect(config=CFG_S, timeout=3))
    Soniox.url = old
    check("refused port -> StreamError(network), transient",
          e is not None and e.kind == "network" and not e.fatal, e and (e.kind, str(e)))

    ms.reset([{"reject": soniox_error(401, "unauthenticated", "Incorrect API key provided.")}])
    t0 = time.monotonic()
    e = error_of(lambda: stream.connect(config=CFG_S))
    check("soniox bad key (error frame in the grace) -> auth, fatal, from connect()",
          e is not None and (e.kind, e.fatal, e.code, str(e)) == ("auth", True, 401, "Incorrect API key provided."),
          e and (e.kind, str(e)))
    check("... and it took about the grace, not the timeout", time.monotonic() - t0 < 1.5)

    ms.reset([{"drop_start": True}])
    e = error_of(lambda: stream.connect(config=CFG_S))
    check("soniox drops the connection in the grace -> network", e is not None and e.kind == "network", e)

    ms.reset([{"http": 401}])
    e = error_of(lambda: stream.connect(config=CFG_S))
    check("HTTP 401 on the upgrade -> auth, fatal", e is not None and (e.kind, e.fatal, e.code) == ("auth", True, 401), e)
    ms.reset([{"http": 503}])
    e = error_of(lambda: stream.connect(config=CFG_S))
    check("HTTP 503 on the upgrade -> server, transient", e is not None and (e.kind, e.fatal) == ("server", False), e)

    ma.reset(["reject:1008:Unauthorized Connection: Invalid API key"])
    e = error_of(lambda: stream.connect(config=CFG_A))
    check("assemblyai bad key -> auth, fatal, provider's message",
          e is not None and (e.kind, e.fatal, e.code, str(e)) == ("auth", True, 1008, "Unauthorized Connection: Invalid API key"),
          e and (e.kind, str(e)))
    ma.reset(["reject:1008:Unauthorized Connection: Too many concurrent sessions"])
    e = error_of(lambda: stream.connect(config=CFG_A))
    check("assemblyai too many sessions -> rate_limit, retry_after 60",
          e is not None and (e.kind, e.fatal, e.retry_after) == ("rate_limit", False, 60), e and e.kind)
    ma.reset(["nobegin"])
    t0 = time.monotonic()
    e = error_of(lambda: stream.connect(config=CFG_A, timeout=0.5))
    check("assemblyai without Begin -> timeout kind, transient, within the timeout",
          e is not None and (e.kind, e.fatal) == ("timeout", False) and time.monotonic() - t0 < 1.5, e and e.kind)


def test_keepalive(ms):
    section("keepalive")
    old = stream.KEEPALIVE_SECONDS
    stream.KEEPALIVE_SECONDS = 0.3
    try:
        ms.reset([{}])
        sess = stream.connect(config=CFG_S)
        time.sleep(1.05)
        rec = ms.sessions[0]
        idle = ms.keepalives(rec)
        check("idle: {\"type\": \"keepalive\"} every interval", idle in (3, 4), idle)
        for _ in range(8):
            sess.send(silence(0.1))
            time.sleep(0.1)
        busy = ms.keepalives(rec) - idle
        check("no keepalive while audio flows", busy == 0, busy)
        sess.finish(2)
        check("no keepalive after the end of audio", rec["texts"][-1] == "", rec["texts"][-3:])
    finally:
        stream.KEEPALIVE_SECONDS = old


def test_terms(ms, ma):
    section("key terms: dedupe, caps, notes")
    many = [f"term{i:05d}" for i in range(4000)]   # 9 characters each
    ms.reset([{}])
    sess = stream.connect(config=CFG_S, terms=many + ["term00000"])
    sent = ms.sessions[0]["start"]["context"]["terms"]
    used = sum(len(t) + 1 for t in sent)
    check("soniox: capped at 9,000 characters of terms (the docs: 8,000 tokens is ~10,000), first kept",
          sent == many[:len(sent)] and used <= 9000 and len(sent) == 900, (len(sent), used))
    check("soniox: the note says how many of how many",
          sess.info.note == f"key terms: sent the first {len(sent)} of 4000 (Soniox's context limit)", sess.info.note)
    sess.close()
    ms.reset([{}])
    sess = stream.connect(config=CFG_S, terms=["Kubernetes", "PostgreSQL", "Kubernetes"])
    check("soniox: duplicates dropped, no note",
          ms.sessions[0]["start"]["context"]["terms"] == ["Kubernetes", "PostgreSQL"] and sess.info.note is None)
    sess.close()

    ma.reset()
    sess = stream.connect(config=CFG_A, terms=[f"t{i}" for i in range(150)] + ["t0"])
    sent = json.loads(ma.sessions[0]["query"]["keyterms_prompt"][0])
    check("assemblyai: the first 100 sent", sent == [f"t{i}" for i in range(100)], len(sent))
    check("assemblyai: the note says 100 of 150",
          sess.info.note == "key terms: sent the first 100 of 150 (AssemblyAI streaming takes at most 100)",
          sess.info.note)
    sess.close()


def test_assemblyai(ma):
    section("assemblyai: handshake, turns, finish")
    ma.reset()
    sess = stream.connect(config=CFG_A, endpoint_ms=2400, languages=["pt"])
    rec = ma.sessions[0]
    q = rec["query"]
    check("raw key as Authorization", rec["auth"] == KEY_A, rec["auth"])
    check("URL: model, rate, terms, max_turn_silence; no format_turns, no languages",
          q == {"speech_model": ["universal-3-6-pro"], "sample_rate": ["16000"], "max_turn_silence": ["2400"],
                "keyterms_prompt": ['["Kubernetes"]']}, q)
    info = sess.info
    check("info: provider, model, limits",
          (info.provider, info.model, info.max_burst_seconds, info.max_session_seconds)
          == ("assemblyai", "universal-3-6-pro", 15, 10800), info)
    for _ in range(25):
        sess.send(silence(0.1))
    events, _ = collect(sess, 2, until=lambda ev: len([e for e in ev if isinstance(e, Final)]) >= 2)
    events += sess.finish(3)
    check("Turn frames -> Partial / Final with stt's own turn numbers and session ms",
          events == [Partial(0, "s0 t", 0), Final(0, "s0 t0", 0, 1000), Partial(1, "s0 t", 1000),
                     Final(1, "s0 t1", 1000, 2000), Final(2, "s0 end", 2000, 2500)],
          "\n        ".join(map(str, events)))
    check("finish sends Terminate", [json.loads(t) for t in rec["texts"]] == [{"type": "Terminate"}], rec["texts"])

    ma.reset(["double"])
    sess = stream.connect(config={**CFG_A, "assemblyai_stream_model": "universal-streaming-english"})
    check("Universal-Streaming asks for format_turns", ma.sessions[0]["query"].get("format_turns") == ["true"])
    for _ in range(10):
        sess.send(silence(0.1))
    events = sess.finish(3)
    check("format_turns: the unformatted copy is a Partial, one Final per turn",
          events == [Partial(0, "s0 t", 0), Partial(0, "s0 t0", 0), Final(0, "S0 t0.", 0, 1000)],
          "\n        ".join(map(str, events)))

    ma.reset(["silent"])
    sess = stream.connect(config=CFG_A)
    sess.send(silence(1.0))
    sess.send(silence(0.3))
    events, _ = collect(sess, 1, until=lambda ev: ev)
    t0 = time.monotonic()
    rest = sess.finish(0.4)
    check("finish timeout: the last Partial comes back as the Final, timed_out set",
          events == [Partial(0, "s0 t", 0)] and rest == [Final(0, "s0 t", 0)] and sess.timed_out,
          (events, rest, sess.timed_out))
    check("finish timeout: within timeout + 1 s", time.monotonic() - t0 < 1.4)


def test_assemblyai_chunks(ma):
    section("assemblyai: chunking")
    ma.reset()
    sess = stream.connect(config=CFG_A)
    audio = bytearray()
    for _ in range(10):                      # 10 ms sends: buffered to 50 ms
        audio += b"\1" * 320
        sess.send(b"\1" * 320)
    big = b"\2" * (int(2.5 * BPS) + 1)      # 2.5 s and an odd byte: split
    audio += big
    sess.send(big)
    audio += b"\3" * 33
    sess.send(b"\3" * 33)
    sess.finish(3)
    rec = ma.sessions[0]
    chunks = rec["chunks"]
    check("every chunk 50-1000 ms and of even length",
          chunks and all(1600 <= n <= BPS and n % 2 == 0 for n in chunks), chunks)
    check("small sends were buffered, big ones split", chunks[:2] == [1600, 1600] and chunks[2:5] == [BPS, BPS, BPS // 2],
          chunks)
    received = bytes(rec["audio"])
    check("the audio arrives unchanged; only the very last chunk is padded with silence",
          received[:len(audio)] == bytes(audio) and set(received[len(audio):]) == {0}
          and len(received) - len(audio) < chunks[-1] and chunks[-1] == 1600, (len(received), len(audio), chunks[-1]))


AAI_ERROR_ROWS = [
    # (behaviour, kind, fatal, routine, retry_after, code)
    ("error:1008:1:Unauthorized Connection: Insufficient funds", "credit", True, False, None, 1008),
    ("error:1008:1:Unauthorized Connection: Invalid API key", "auth", True, False, None, 1008),
    ("error:1008:1:Unauthorized Connection: Too many concurrent sessions", "rate_limit", False, False, 60, 1008),
    ("error:3006:1:Invalid 'keyterms_prompt'", "rejected", True, False, None, 3006),
    ("close:3008:1", "limit", False, True, None, 3008),
    # An unasked Termination leaves it to what follows (as audiorecorder read it).
    ("expire:3008:1", "limit", False, True, None, 3008),
    ("errterm:1008:1:Unauthorized Connection: Insufficient funds", "credit", True, False, None, 1008),
    ("close:1011:1", "server", False, False, None, 1011),
    ("close:3007:1", "server", False, False, None, 3007),
    ("drop:1", "network", False, False, None, None),
]


def test_assemblyai_errors(ma):
    section("assemblyai: errors mid-session")
    for behaviour, kind, fatal, routine, retry, code in AAI_ERROR_ROWS:
        ma.reset([behaviour])
        sess = stream.connect(config=CFG_A)
        for _ in range(12):
            try:
                sess.send(silence(0.1))
            except SessionClosed:
                break
        events, closed = collect(sess, 3)
        e = closed.error if closed else None
        got = e and (e.kind, e.fatal, e.routine, e.retry_after)
        check(f"{behaviour.split(':')[0]} {code or ''} -> {kind}, fatal={fatal}, routine={routine}, retry_after={retry}",
              got == (kind, fatal, routine, retry) and (code is None or e.code == code), got)
        if behaviour.startswith("error:1008:1:Unauthorized Connection: Insufficient"):
            check("the error frame's text wins over the close reason",
                  str(e) == "Unauthorized Connection: Insufficient funds", str(e))
        sess.close()


def test_close(ms, ma):
    section("close")
    for mock, cfg, name in ((ms, CFG_S, "soniox"), (ma, CFG_A, "assemblyai")):
        for rec in mock.sessions:
            rec["done"].wait(2)
        baseline = settle_threads()
        mock.reset()
        sess = stream.connect(config=cfg)
        sess.send(silence(0.3))
        t0 = time.monotonic()
        sess.close()
        took = time.monotonic() - t0
        sess.close()
        check(f"{name}: close() is quick and idempotent", took < 1.0, took)
        check(f"{name}: no session thread left running",
              not sess._thread.is_alive() and not sess._ws.recv_events_thread.is_alive())
        try:
            sess.poll()
            check(f"{name}: poll after close raises SessionClosed", False)
        except SessionClosed as e:
            check(f"{name}: poll after close raises SessionClosed(None)", e.error is None and e.events == [])
        mock.sessions[0]["done"].wait(2)
        now = threads_back_to(baseline)
        check(f"{name}: threading.active_count back to baseline", now <= baseline, (now, baseline))

    baseline = settle_threads()
    sess = stream.connect(config={"provider": "fake"})
    sess.send(silence(1))
    sess.finish()
    sess.close()
    check("fake: no thread at all", threading.active_count() <= baseline, (threading.active_count(), baseline))


def test_stalled(peer):
    section("stalled network")
    old_url, old_stall = Soniox.url, stream.STALL_SECONDS
    Soniox.url = peer.url
    try:
        # Audio the network stopped taking: the pings cannot go out (the
        # blocked send holds the connection's lock); TCP_USER_TIMEOUT fails it.
        stream.STALL_SECONDS = 1
        sess = stream.connect(config=CFG_S)
        out, took = bounded(lambda: send_forever(sess), 10, sess)
        check("send() into a stalled network fails within a few s: SessionClosed(network)",
              isinstance(out, SessionClosed) and out.error is not None and out.error.kind == "network"
              and took < 6, (repr(out), round(took, 1)))
        sess.close()

        stream.STALL_SECONDS = 30
        sess = stream.connect(config=CFG_S)
        sess.send(silence(0.1))
        fill_send_buffer(sess)
        out, took = bounded(lambda: sess.finish(0.5), 5, sess)
        check("finish(0.5) with the send buffer full: back within timeout + 1 s, timed_out set",
              isinstance(out, list) and took < 1.5 and sess.timed_out, (repr(out), round(took, 1), sess.timed_out))
        sess.close()

        sess = stream.connect(config=CFG_S)
        cut = threading.Timer(0.5, lambda: sess.abort())
        cut.start()
        out, took = bounded(lambda: send_forever(sess), 5, sess)
        cut.join()
        check("abort() from another thread frees a blocked send at once: SessionClosed(None)",
              isinstance(out, SessionClosed) and out.error is None and took < 1.5, (repr(out), round(took, 1)))
        sess.close()
    finally:
        Soniox.url, stream.STALL_SECONDS = old_url, old_stall


def test_fake():
    section("fake provider")
    cfg = {"provider": "fake", "keyterms": "a,b,a", "languages": "pt,en"}
    sess = stream.connect(config=cfg, speakers=True)
    check("info", (sess.info.provider, sess.info.model, sess.info.max_burst_seconds, sess.info.max_session_seconds)
          == ("fake", "fake", 15, 3600))
    sess.send(silence(1.25))
    sess.send(silence(1.0))
    events = sess.poll()
    events += sess.finish()
    check("partials every 0.5 s, a final every 2 s, finish names totals",
          events == [Partial(0, "fake turn 0 word", 0), Partial(0, "fake turn 0 word word", 0),
                     Partial(0, "fake turn 0 word word word", 0), Final(0, "fake turn 0 bytes=64000", 0, 2000, "A", "pt"),
                     Final(1, "fake turn 1 end total=72000 terms=2 languages=pt,en", 2000, 2250, "A", "pt")],
          "\n        ".join(map(str, events)))

    sess = stream.connect(config={**cfg, "fake_fail": "after:1:network"})
    sess.send(silence(0.75))
    sess.send(silence(0.75))
    try:
        sess.send(silence(0.1))
        check("after:1:network ends the session after 1 s of audio", False)
    except SessionClosed as e:
        check("after:1:network ends the session after 1 s of audio, events first, the open turn final",
              e.error.kind == "network" and not e.error.fatal
              and e.events == [Partial(0, "fake turn 0 word", 0), Partial(0, "fake turn 0 word word", 0),
                               Final(0, "fake turn 0 word word", 0, 1000, None, "pt")], (e.error, e.events))

    for kind, fatal, routine, retry in (("auth", True, False, None), ("limit", False, True, None),
                                        ("rate_limit", False, False, 1.0), ("config", True, False, None)):
        e = error_of(lambda: stream.connect(config={**cfg, "fake_fail": f"connect:{kind}"}))
        check(f"connect:{kind} -> fatal={fatal}, routine={routine}, retry_after={retry}",
              e is not None and (e.kind, e.fatal, e.routine, e.retry_after) == (kind, fatal, routine, retry), e)

    sess = stream.connect(config={**cfg, "fake_finish_delay": "0.2"})
    sess.send(silence(0.6))
    check("finish delay shorter than the timeout: normal end",
          [e.text for e in sess.finish(2)][-1].startswith("fake turn 0 end") and sess.timed_out is None)
    sess = stream.connect(config={**cfg, "fake_finish_delay": "5"})
    sess.send(silence(0.6))
    t0 = time.monotonic()
    rest = sess.finish(0.3)
    check("finish delay past the timeout: best-effort Final, timed_out, no hang",
          rest == [Partial(0, "fake turn 0 word", 0), Final(0, "fake turn 0 word", 0, 600, None, "pt")]
          and sess.timed_out and time.monotonic() - t0 < 1, rest)
    t0 = time.monotonic()
    e = error_of(lambda: stream.connect(config={**cfg, "fake_connect_delay": "5"}, timeout=0.3))
    check("connect delay past the timeout -> timeout kind", e is not None and e.kind == "timeout"
          and time.monotonic() - t0 < 1, e)


# --------------------------------------------------------------------------
# tests: `stt stream`
# --------------------------------------------------------------------------

class Child:
    """A `stt stream` process; its stdout lines are read as they come."""

    def __init__(self, argv, config_path):
        env = {**os.environ, "STT_CONFIG": config_path, "PYTHONDONTWRITEBYTECODE": "1"}
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=env, cwd=ROOT)
        self.lines = queue.Queue()
        self.seen = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.p.stdout:
            try:
                self.lines.put(json.loads(raw))
            except ValueError:
                self.lines.put({"type": "NOT JSON", "raw": raw.decode(errors="replace")})
        self.lines.put(None)

    def next(self, timeout=3):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is not None:
            self.seen.append(line)
        return line

    def until(self, pred, timeout=3):
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            line = self.next(left)
            if line is None:
                return None
            if pred(line):
                return line
        return None

    def write(self, data):
        try:
            self.p.stdin.write(data)
            self.p.stdin.flush()
        except BrokenPipeError:   # it exited already; finish() says why
            pass

    def close_stdin(self):
        try:
            self.p.stdin.close()
        except OSError:   # it exited with audio still buffered
            pass

    def finish(self, timeout=5):
        """Wait for exit; returns (status, every stdout line, stderr)."""
        self.close_stdin()
        try:
            status = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            status = "hung"
        while (line := self.next(1)) is not None:
            pass
        return status, self.seen, self.p.stderr.read().decode(errors="replace")


def runner(soniox_url=None, backlog=None, ping=None):
    """argv running `stt stream` (as `python3 -m stt.stream_cli`, or through
    a -c wrapper that points Soniox at the mock and sets test timings)."""
    if soniox_url is None and backlog is None and ping is None:
        return [sys.executable, "-m", "stt.stream_cli"]
    code = f"import sys; sys.path.insert(0, {ROOT!r})\n"
    if soniox_url:
        code += (f"from stt.stream import soniox; soniox.Soniox.url = {soniox_url!r}; "
                 f"soniox.ACCEPT_GRACE_SECONDS = {GRACE}\n")
    if ping is not None:
        code += f"from stt import stream; stream.PING_SECONDS = {ping}\n"
    code += "from stt import stream_cli\n"
    if backlog is not None:
        code += f"stream_cli.BACKLOG_SECONDS = {backlog}\n"
    code += "sys.exit(stream_cli.main(sys.argv[1:]))"
    return [sys.executable, "-c", code]


def last_stderr_line(stderr):
    lines = [line for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def run_stdout_closed(argv, config_path):
    """Run `stt stream` with stdin empty and stdout a pipe nobody reads any
    more. Returns (status, stderr)."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        r = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=write_end, stderr=subprocess.PIPE, cwd=ROOT,
                           env={**os.environ, "STT_CONFIG": config_path, "PYTHONDONTWRITEBYTECODE": "1"},
                           timeout=10)
    finally:
        os.close(write_end)
    return r.returncode, r.stderr.decode(errors="replace")


def test_cli(work, ms):
    section("stt stream")
    fake = os.path.join(work, "fake.conf")
    write_config(fake, provider="fake", keyterms="Kubernetes, PostgreSQL", languages="pt,en")

    c = Child(runner() + ["--languages", "en"], fake)
    ready = c.next()
    check("ready first, with provider and model", ready == {"type": "ready", "provider": "fake", "model": "fake"}, ready)
    c.write(silence(1.0))
    first = c.next()
    second = c.next()
    check("partials arrive while stdin is still open (flushed per line)",
          first == {"type": "partial", "turn": 0, "text": "fake turn 0 word", "start_ms": 0}
          and second and second["text"] == "fake turn 0 word word", (first, second))
    c.write(silence(1.0) + silence(0.3))
    final = c.until(lambda line: line["type"] == "final")
    check("final line: all fields", final == {"type": "final", "turn": 0, "text": "fake turn 0 bytes=64000",
                                              "start_ms": 0, "end_ms": 2000, "speaker": None, "language": "en"}, final)
    c.close_stdin()
    status, lines, stderr = c.finish()
    types = [line["type"] for line in lines]
    check("stdin EOF: last final, then end, exit 0",
          status == 0 and types[-2:] == ["final", "end"]
          and lines[-2]["text"] == "fake turn 1 end total=73600 terms=2 languages=en", (status, lines[-2:], stderr))
    check("order: ready, then partials/finals by turn, end last",
          types[0] == "ready" and types.count("end") == 1
          and [line["turn"] for line in lines if "turn" in line] == sorted(line["turn"] for line in lines if "turn" in line),
          types)
    check("nothing but JSON lines on stdout, nothing on stderr", "NOT JSON" not in types and stderr == "", stderr)

    terms_file = os.path.join(work, "terms.txt")
    with open(terms_file, "w") as f:
        f.write("# names\nSão Paulo, SP\nKubernetes\n")
    s_conf = os.path.join(work, "soniox.conf")
    write_config(s_conf, provider="soniox", soniox_api_key=KEY_S, languages="pt,en,de")
    ms.reset([{"auto": True}])
    c = Child(runner(ms.url) + ["--terms-file", terms_file, "--languages", "en", "--speakers",
                                "--endpoint-ms", "1200"], s_conf)
    check("soniox: ready", c.next() == {"type": "ready", "provider": "soniox", "model": "stt-rt-v5"})
    rec = ms.sessions[0] if ms.sessions else {}
    check("soniox: bearer key and the flags reach the start message",
          rec.get("auth") == f"Bearer {KEY_S}" and rec["start"]["context"] == {"terms": ["São Paulo, SP", "Kubernetes"]}
          and rec["start"]["language_hints"] == ["en"] and rec["start"]["enable_speaker_diarization"] is True
          and rec["start"]["max_endpoint_delay_ms"] == 1200, rec.get("start"))
    for _ in range(5):
        c.write(silence(0.1))
    partial = c.until(lambda line: line["type"] == "partial")
    check("soniox: a partial while stdin is open", partial and partial["text"].startswith("w0"), partial)
    for _ in range(6):
        c.write(silence(0.1))
    final = c.until(lambda line: line["type"] == "final")
    check("soniox: a turn's final", final and final["text"] == "w0 w1 w2 w3" and final["language"] == "en"
          and final["end_ms"] == 1000, final)
    for _ in range(8):   # 1.9 s in all: w4-w6 still open at EOF
        c.write(silence(0.1))
    c.until(lambda line: line["type"] == "partial" and line["text"] == "w4 w5 w6")
    c.close_stdin()
    status, lines, stderr = c.finish()
    check("soniox: EOF -> remaining final, end, exit 0",
          status == 0 and [line["type"] for line in lines[-2:]] == ["final", "end"]
          and lines[-2]["text"] == "w4 w5 w6", (status, lines[-3:], stderr))
    check("soniox: the empty text frame ended the stream", rec["texts"] == [""], rec.get("texts"))

    # SIGTERM / SIGINT: cancel at once, nothing more on stdout.
    for signum, expected in ((signal.SIGTERM, 143), (signal.SIGINT, 130)):
        c = Child(runner(), fake)
        c.next()
        c.write(silence(0.6))
        c.next()
        time.sleep(0.2)
        t0 = time.monotonic()
        c.p.send_signal(signum)
        status, lines, stderr = c.finish(3)
        took = time.monotonic() - t0
        check(f"{signal.Signals(signum).name}: exit {expected} in < 1 s, nothing more printed",
              status == expected and took < 1 and [line["type"] for line in lines] == ["ready", "partial"]
              and stderr == "", (status, took, lines, stderr))

    slow = os.path.join(work, "slow.conf")
    write_config(slow, provider="fake", fake_finish_delay=5)
    c = Child(runner(), slow)
    c.next()
    c.write(silence(0.6))
    c.next()
    c.close_stdin()
    time.sleep(0.3)
    t0 = time.monotonic()
    c.p.send_signal(signal.SIGTERM)
    status, lines, stderr = c.finish(3)
    check("SIGTERM during a slow finish: exit 143 in < 1 s, no end line",
          status == 143 and time.monotonic() - t0 < 1 and "end" not in [line["type"] for line in lines],
          (status, lines, stderr))

    ms.reset([{"auto": True}])
    c = Child(runner(ms.url), s_conf)
    c.next()
    c.write(silence(0.6))
    c.until(lambda line: line["type"] == "partial")
    t0 = time.monotonic()
    c.p.send_signal(signal.SIGTERM)
    status, lines, stderr = c.finish(3)
    check("soniox SIGTERM: exit 143 in < 1 s and the provider connection is closed",
          status == 143 and time.monotonic() - t0 < 1 and ms.sessions[0]["done"].wait(1)
          and "" not in ms.sessions[0]["texts"], (status, stderr))

    # Finish timeout: best effort, a note, end, exit 0.
    c = Child(runner() + ["--finish-timeout", "0.4"], slow)
    c.next()
    c.write(silence(1.1))
    c.close_stdin()
    t0 = time.monotonic()
    status, lines, stderr = c.finish(4)
    check("finish timeout: last turn as last heard, a note, end, exit 0",
          status == 0 and [line["type"] for line in lines[-3:]] == ["final", "note", "end"]
          and lines[-3]["text"] == "fake turn 0 word word" and "not confirmed" in lines[-2]["message"]
          and time.monotonic() - t0 < 2, (status, lines[-3:], stderr))

    # Audio written while connecting is kept.
    delayed = os.path.join(work, "delayed.conf")
    write_config(delayed, provider="fake", fake_connect_delay=1, languages="")
    c = Child(runner(), delayed)
    c.write(silence(1.5) + b"\0" * 7)
    c.close_stdin()
    status, lines, stderr = c.finish(5)
    check("audio written before ready is not lost",
          status == 0 and lines[0]["type"] == "ready"
          and any(line.get("text", "").startswith(f"fake turn 0 end total={int(1.5 * BPS) + 7} ") for line in lines),
          (status, lines, stderr))

    # Failures: error line on stdout, `stt: ...` last on stderr, exit status by class.
    no_key = os.path.join(work, "nokey.conf")
    write_config(no_key, provider="soniox")
    ms.reset()
    c = Child(runner(ms.url), no_key)
    status, lines, stderr = c.finish(5)
    check("config problem: exit 3, error line kind config, before any network",
          status == 3 and lines == [{"type": "error", "kind": "config", "fatal": True, "routine": False,
                                     "retry_after": None,
                                     "message": f"soniox_api_key is not set in {no_key}"}]
          and ms.sessions == [], (status, lines, ms.sessions))
    check("config problem: `stt: <message>` is the last stderr line",
          last_stderr_line(stderr) == f"stt: soniox_api_key is not set in {no_key}", stderr)
    c = Child(runner() + ["--provider", "whisper"], fake)
    status, lines, _ = c.finish(5)
    check("unknown provider: exit 3", status == 3 and lines[0]["kind"] == "config", (status, lines))
    c = Child(runner() + ["--terms-file", os.path.join(work, "missing.txt")], fake)
    status, lines, _ = c.finish(5)
    check("unreadable --terms-file: exit 3", status == 3 and lines[0]["kind"] == "config", (status, lines))
    not_utf8 = os.path.join(work, "latin1.txt")
    with open(not_utf8, "wb") as f:
        f.write("São Paulo\n".encode("latin-1"))
    c = Child(runner() + ["--terms-file", not_utf8], fake)
    status, lines, _ = c.finish(5)
    check("--terms-file that is not UTF-8: exit 3, not an internal error",
          status == 3 and lines[0]["kind"] == "config", (status, lines))

    ms.reset([{"reject": soniox_error(401, "unauthenticated", f"Incorrect API key provided: {KEY_S}")}])
    c = Child(runner(ms.url), s_conf)
    status, lines, stderr = c.finish(5)
    check("fatal provider error: exit 4, error line, key scrubbed",
          status == 4 and lines == [{"type": "error", "kind": "auth", "fatal": True, "routine": False,
                                     "retry_after": None, "message": "Incorrect API key provided: ***"}],
          (status, lines))
    check("fatal: last stderr line, no key anywhere",
          last_stderr_line(stderr) == "stt: Incorrect API key provided: ***" and KEY_S not in stderr, stderr)

    failing = os.path.join(work, "failing.conf")
    write_config(failing, provider="fake", fake_fail="after:1:network")
    c = Child(runner(), failing)
    c.next()
    c.write(silence(1.5))
    status, lines, stderr = c.finish(5)
    types = [line["type"] for line in lines]
    check("transient error mid-session: partials, the open turn's final, then the error line, exit 5",
          status == 5 and types == ["ready", "partial", "partial", "final", "error"]
          and lines[-2]["text"] == "fake turn 0 word word"
          and lines[-1]["kind"] == "network" and lines[-1]["fatal"] is False, (status, lines))
    check("transient: `stt: ...` last on stderr", last_stderr_line(stderr).startswith("stt: fake network failure"), stderr)

    ms.reset([{"drop_after": 2}])
    c = Child(runner(ms.url), s_conf)
    c.next()
    for _ in range(3):
        c.write(silence(0.1))
        time.sleep(0.05)
    status, lines, stderr = c.finish(5)
    check("soniox connection dropped: exit 5, kind network",
          status == 5 and lines[-1]["type"] == "error" and lines[-1]["kind"] == "network", (status, lines[-1:]))

    port = free_port()
    c = Child(runner(f"ws://127.0.0.1:{port}/x"), s_conf)
    status, lines, stderr = c.finish(5)
    check("provider unreachable: exit 5", status == 5 and lines and lines[0]["kind"] == "network", (status, lines))

    for kind, expected in (("rejected", 4), ("limit", 5), ("internal", 1)):
        conf = os.path.join(work, f"connect-{kind}.conf")
        write_config(conf, provider="fake", fake_fail=f"connect:{kind}")
        status, lines, _ = Child(runner(), conf).finish(5)
        check(f"connect:{kind} -> exit {expected}", status == expected and lines[0]["kind"] == kind, (status, lines))

    backlog = os.path.join(work, "backlog.conf")
    write_config(backlog, provider="fake", fake_connect_delay=1.5)
    c = Child(runner(backlog=0.5), backlog)
    c.write(silence(1.0))
    status, lines, stderr = c.finish(5)
    check("more audio piled up than the backlog allows: exit 1, kind internal",
          status == 1 and lines[-1]["type"] == "error" and lines[-1]["kind"] == "internal", (status, lines, stderr))

    ms.reset([{}])
    c = Child(runner(ms.url), s_conf)
    status, lines, stderr = c.finish(5)
    check("soniox: stdin closed before any audio -> ready, end, exit 0 (no 'No audio received.')",
          status == 0 and [line["type"] for line in lines] == ["ready", "end"] and stderr == "", (status, lines, stderr))

    unreadable = os.path.join(work, "config-dir")
    os.makedirs(unreadable, exist_ok=True)
    status, lines, stderr = Child(runner(), unreadable).finish(5)
    check("unreadable config: exit 3, error line kind config, `stt: cannot read ...` last on stderr",
          status == 3 and lines and lines[0]["kind"] == "config"
          and last_stderr_line(stderr).startswith(f"stt: cannot read {unreadable}"), (status, lines, stderr))

    status, stderr = run_stdout_closed(runner(), no_key)
    check("stdout closed, config error: exit 3, `stt: ...` last on stderr, no traceback",
          status == 3 and last_stderr_line(stderr) == f"stt: soniox_api_key is not set in {no_key}"
          and "Traceback" not in stderr and "Exception ignored" not in stderr, (status, stderr))
    status, stderr = run_stdout_closed(runner(), fake)
    check("stdout closed mid-session: exit 1, `stt: stdout closed`, no traceback",
          status == 1 and last_stderr_line(stderr) == "stt: stdout closed" and "Traceback" not in stderr
          and "Exception ignored" not in stderr, (status, stderr))


def test_cli_stalled(work, peer):
    section("stt stream on a stalled network")
    conf = os.path.join(work, "stalled.conf")
    write_config(conf, provider="soniox", soniox_api_key=KEY_S, languages="en")

    # The main loop sits in a send on a full buffer: the signal must still win.
    c = Child(runner(peer.url, backlog=600), conf)
    ready = c.next()
    feeder = threading.Thread(target=c.write, args=(silence(100),), daemon=True)   # 3.2 MB
    feeder.start()
    time.sleep(1.5)
    t0 = time.monotonic()
    c.p.send_signal(signal.SIGTERM)
    status, lines, stderr = c.finish(4)
    took = time.monotonic() - t0
    check("SIGTERM while a send is blocked: exit 143 in < 1 s, nothing more printed",
          ready and ready["type"] == "ready" and status == 143 and took < 1
          and [line["type"] for line in lines] == ["ready"] and stderr == "", (status, round(took, 2), lines, stderr))

    # A keepalive ping that finds the network dead: one error line, and on
    # stderr only `stt: ...` (websockets' own traceback stays out).
    c = Child(runner(peer.url, ping=0.3), conf)
    c.next()
    error = c.until(lambda line: line["type"] == "error", timeout=6)
    status, lines, stderr = c.finish(4)
    check("dead network found by a ping: error line kind network, exit 5",
          error is not None and error["kind"] == "network" and status == 5, (status, lines))
    check("... and stderr holds only the `stt: ...` line",
          [line for line in stderr.splitlines() if line.strip()] == [f"stt: {error and error['message']}"], stderr)


def main():
    work = tempfile.mkdtemp(prefix="stt-stream-test-")
    os.environ["STT_CONFIG"] = os.path.join(work, "no-such-config")   # never the real one
    soniox_module.ACCEPT_GRACE_SECONDS = GRACE
    ms, ma, peer = MockSoniox(), MockAAI(), StalledPeer()
    Soniox.url, AssemblyAI.url = ms.url, ma.url
    try:
        test_imports()
        test_config_api(ms)
        test_soniox_handshake(ms)
        test_soniox_turns(ms)
        test_soniox_finish(ms)
        test_soniox_errors(ms)
        test_connect_failures(ms, ma)
        test_keepalive(ms)
        test_terms(ms, ma)
        test_assemblyai(ma)
        test_assemblyai_chunks(ma)
        test_assemblyai_errors(ma)
        test_close(ms, ma)
        test_stalled(peer)
        test_fake()
        test_cli(work, ms)
        test_cli_stalled(work, peer)
    finally:
        ms.stop()
        ma.stop()
        peer.stop()
        subprocess.run(["rm", "-rf", work])
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
