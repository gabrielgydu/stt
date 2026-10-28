#!/usr/bin/env python3
"""
Exercise both providers, the fake one and the command against a local mock of
their HTTP APIs: no network, no credit, and never the real config.

    python3 tests/test_stt.py
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import stt  # noqa: E402
from stt import config, spelling  # noqa: E402
from stt.providers.assemblyai import AssemblyAI  # noqa: E402
from stt.providers.soniox import Soniox, to_transcript  # noqa: E402

failures = []
checks = 0


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


def wait_for(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------
# mock provider APIs
# --------------------------------------------------------------------------

SONIOX_TOKENS = [
    {"text": "Bom", "speaker": "1", "language": "pt", "start_ms": 0, "end_ms": 200},
    {"text": " dia,", "speaker": "1", "language": "pt", "start_ms": 200, "end_ms": 400},
    {"text": " Ana.", "speaker": "1", "language": "pt", "start_ms": 400, "end_ms": 900},
    {"text": " Hi", "speaker": "2", "language": "en", "start_ms": 1000, "end_ms": 1200},
    {"text": " there.", "speaker": "2", "language": "en", "start_ms": 1200, "end_ms": 1500},
    {"text": " Tudo", "speaker": "1", "language": "pt", "start_ms": 1600, "end_ms": 1800},
    {"text": " bem?", "speaker": "1", "language": "pt", "start_ms": 1800, "end_ms": 2000},
]

AAI_DONE = {
    "id": "aai-1", "status": "completed", "text": "Bom dia. Hi there.", "language_code": "pt",
    "utterances": [
        {"speaker": "A", "text": "Bom dia.", "start": 0, "end": 900},
        {"speaker": "B", "text": "Hi there.", "start": 1000, "end": 1500},
    ],
    "words": [{"text": "Bom", "start": 0, "end": 300}, {"text": "there.", "start": 1200, "end": 1500}],
}


class Mock:
    """State shared with the handler; tests set the knobs and read the log."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.requests = []         # (method, path, headers, body)
        self.polls_until_done = 1  # status answers "processing" this many times first
        self.fail_job = False
        self.fail_state = "error"  # Soniox's word for a failed job: "error" or "failed"
        self.poll_status = None    # force an HTTP status on every status check
        self.poll_fail_count = 0   # ... or only on the first N
        self.create_status = None  # force an HTTP status on job creation
        self.create_body = None    # answer job creation with this instead
        self.error_body = None     # body of forced errors (dict, or bytes as is)
        self.reason = None         # HTTP reason phrase of forced errors
        self.job_error = "audio decode failed"
        self.script = {}           # (method, path) -> [statuses], one popped per request
        self.drop = {}             # (method, path) -> times to take the request and hang up unanswered
        self.delete_status = {}    # path -> forced status for DELETE
        self.delete_gate = None    # threading.Event: DELETEs wait for it
        self.deletes_answered = 0
        self.tokens = SONIOX_TOKENS
        self.aai_result = AAI_DONE
        self.polls = 0
        self.poll_times = []


MOCK = Mock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, status, payload=None, reason=None):
        if isinstance(payload, bytes):
            body = payload
        else:
            body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status, reason)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forced(self, status):
        return self._reply(status, MOCK.error_body if MOCK.error_body is not None else {"error_message": "forced"},
                           MOCK.reason)

    def _record(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        MOCK.requests.append((self.command, self.path, dict(self.headers), body))
        return body

    def _dropped(self):
        """Hang up without answering when MOCK.drop says so: the provider got
        the request (and may have acted on it), the client never hears."""
        key = (self.command, self.path)
        if not MOCK.drop.get(key):
            return False
        MOCK.drop[key] -= 1
        self.close_connection = True
        self.connection.shutdown(socket.SHUT_RDWR)
        return True

    def _scripted(self):
        statuses = MOCK.script.get((self.command, self.path))
        return statuses.pop(0) if statuses else None

    def _status_payload(self):
        MOCK.polls += 1
        MOCK.poll_times.append(time.monotonic())
        if MOCK.poll_status:
            return MOCK.poll_status, None
        if MOCK.polls <= MOCK.poll_fail_count:
            return 503, None
        if MOCK.fail_job:
            return 200, None
        if MOCK.polls <= MOCK.polls_until_done:
            return 200, "processing"
        return 200, "completed"

    def do_POST(self):
        self._record()
        if self._dropped():
            return None
        forced = self._scripted()
        if forced:
            return self._forced(forced)
        if self.path == "/v1/files":
            return self._reply(201, {"id": "file-1"})
        if self.path == "/v1/transcriptions":
            if MOCK.create_status:
                return self._forced(MOCK.create_status)
            if MOCK.create_body is not None:
                return self._reply(201, MOCK.create_body)
            return self._reply(201, {"id": "tr-1", "status": "queued"})
        if self.path == "/v2/upload":
            return self._reply(200, {"upload_url": "https://cdn.example/audio"})
        if self.path == "/v2/transcript":
            if MOCK.create_status:
                return self._forced(MOCK.create_status)
            return self._reply(200, {"id": "aai-1", "status": "queued"})
        return self._reply(404, {"error": f"no route {self.path}"})

    def do_GET(self):
        self._record()
        if self.path == "/v1/transcriptions/tr-1":
            code, state = self._status_payload()
            if code != 200:
                return self._forced(code)
            if state is None:
                return self._reply(200, {"id": "tr-1", "status": MOCK.fail_state,
                                         "error_message": MOCK.job_error})
            return self._reply(200, {"id": "tr-1", "status": state})
        if self.path == "/v1/transcriptions/tr-1/transcript":
            return self._reply(200, {"id": "tr-1", "text": "", "tokens": MOCK.tokens})
        if self.path == "/v2/transcript/aai-1":
            code, state = self._status_payload()
            if code != 200:
                return self._forced(code)
            if state is None:
                return self._reply(200, {"id": "aai-1", "status": "error", "error": MOCK.job_error})
            if state != "completed":
                return self._reply(200, {"id": "aai-1", "status": state})
            return self._reply(200, MOCK.aai_result)
        return self._reply(404, {"error": f"no route {self.path}"})

    def do_DELETE(self):
        self._record()
        if MOCK.delete_gate is not None:
            MOCK.delete_gate.wait(10)
        MOCK.deletes_answered += 1
        forced = MOCK.delete_status.get(self.path)
        if forced:
            return self._forced(forced)
        return self._reply(204)


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A client that gave up (signal, timeout) leaves the handler writing
        # into a closed socket; that is the test working, not a failure.
        if not isinstance(sys.exc_info()[1], OSError):
            super().handle_error(request, client_address)


def start_mock():
    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    Soniox.base_url = base
    AssemblyAI.base_url = base
    Soniox.poll_seconds = 0.01
    AssemblyAI.poll_seconds = 0.01
    Soniox.submit_retry_waits = (0.01, 0.02)
    AssemblyAI.submit_retry_waits = (0.01, 0.02)
    stt.POLL_ERROR_WINDOW = 0.3
    return server, base


def write_config(path, **values):
    with open(path, "w") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def calls(method=None):
    return [(m, p) for m, p, _, _ in MOCK.requests if method is None or m == method]


def json_body(path):
    for m, p, _, body in MOCK.requests:
        if m == "POST" and p == path:
            return json.loads(body)
    return None


def raised(fn):
    """The exception `fn()` raises, or None."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    return None


SONIOX_DELETES = {("DELETE", "/v1/transcriptions/tr-1"), ("DELETE", "/v1/files/file-1")}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# the command, in a child process
# --------------------------------------------------------------------------

# The command runs in a child process, so the mock's base URL goes in through
# a tiny wrapper rather than the class attribute. STT_TEST_SETUP holds extra
# statements for one run (a lower max_seconds, a shorter window...).
RUNNER = (
    "import os, sys; sys.path.insert(0, %r);"
    "import stt; from stt.providers.soniox import Soniox; from stt.providers.assemblyai import AssemblyAI;"
    "Soniox.base_url = AssemblyAI.base_url = %r;"
    "Soniox.poll_seconds = AssemblyAI.poll_seconds = 0.01;"
    "Soniox.submit_retry_waits = AssemblyAI.submit_retry_waits = (0.01, 0.02);"
    "exec(os.environ.get('STT_TEST_SETUP', ''));"
    "from stt.cli import main; sys.exit(main(sys.argv[1:]))"
)


class Cli:
    def __init__(self, base, cfg_path):
        self.argv = [sys.executable, "-c", RUNNER % (ROOT, base)]
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("STT_")}
        self.env["STT_CONFIG"] = cfg_path

    def run(self, *args, setup="", env=None, stdin=None, text=True, stdout=subprocess.PIPE):
        return subprocess.run(
            self.argv + list(args), stdout=stdout, stderr=subprocess.PIPE, text=text, input=stdin,
            timeout=60, env={**self.env, "STT_TEST_SETUP": setup, **(env or {})},
        )

    def popen(self, *args, setup=""):
        return subprocess.Popen(
            self.argv + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**self.env, "STT_TEST_SETUP": setup},
        )


def last_line(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def failed_cleanly(r, status, needle=""):
    """Exit `status`, last stderr line "stt: ..." containing `needle`, one
    line of ~300 characters, nothing on stdout."""
    line = last_line(r.stderr)
    return (r.returncode == status and line.startswith("stt: ") and needle in line
            and len(line) <= 310 and r.stdout == "")


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_config(work):
    section("config")
    cfg_path = os.path.join(work, "empty")
    os.environ["STT_CONFIG"] = cfg_path
    check("STT_CONFIG sets the path", config.path() == cfg_path)
    check("missing file -> empty config", config.load() == {})
    check("default provider is soniox", config.provider_name({}) == "soniox")
    check("default languages", config.languages({}) == ["pt", "en", "de"])
    check("keyterms split/strip", config.keyterms({"keyterms": "a, b ,,c"}) == ["a", "b", "c"])
    check("no key -> problem names the key and the file",
          stt.config_problem() == f"soniox_api_key is not set in {cfg_path}", stt.config_problem())
    check("unknown provider -> problem",
          "unknown provider 'whisper'" in (stt.config_problem({"provider": "whisper"}) or ""))
    check("key set -> no problem", stt.config_problem({"soniox_api_key": "k"}) is None)
    check("assemblyai needs its own key",
          stt.config_problem({"provider": "assemblyai", "soniox_api_key": "k"})
          == f"assemblyai_api_key is not set in {cfg_path}")
    check("max_seconds follows the provider",
          stt.max_seconds({"soniox_api_key": "k"}) == 5 * 3600
          and stt.max_seconds({"provider": "assemblyai", "assemblyai_api_key": "k"}) == 10 * 3600)
    check("api_keys lists every *_api_key value",
          sorted(config.api_keys({"soniox_api_key": "a", "assemblyai_api_key": "b", "fake_api_key": "",
                                  "keyterms": "x"})) == ["a", "b"])
    check("import stt does not import stt.stream or websockets",
          subprocess.run([sys.executable, "-c",
                          f"import sys; sys.path.insert(0, {ROOT!r}); import stt, stt.cli;"
                          "sys.exit('stt.stream' in sys.modules or 'websockets' in sys.modules)"]).returncode == 0)


def test_spelling():
    section("spellings")
    rules = spelling.Spellings([
        {"from": ["Sao Paulo", "Saopaulo"], "to": "São Paulo"},
        {"from": ["Parana"], "to": "Paraná"},
        {"from": ["Sao Paulo Parana"], "to": "São Paulo e Paraná"},
        {"from": ["Rio"], "to": "Rio de Janeiro"},
        {"from": "SPPR", "to": "Sao Paulo Parana"},
    ])
    check("case-insensitive", rules.apply("sao paulo e SAOPAULO") == "São Paulo e São Paulo",
          rules.apply("sao paulo e SAOPAULO"))
    check("longest form first",
          rules.apply("Sao Paulo Parana disse") == "São Paulo e Paraná disse", rules.apply("Sao Paulo Parana disse"))
    check("whitespace inside a form matches any run", rules.apply("Sao  Paulo") == "São Paulo")
    check("punctuation is a boundary", rules.apply("(Parana), Parana.") == "(Paraná), Paraná.")
    check("no partial-word matches",
          rules.apply("Paranagua SParana Saopaulos 2Parana Parana2") == "Paranagua SParana Saopaulos 2Parana Parana2",
          rules.apply("Paranagua SParana Saopaulos 2Parana Parana2"))
    check("accented neighbours are letters too",
          rules.apply("Paraná") == "Paraná" and rules.apply("áParana") == "áParana"
          and rules.apply("Parana" + chr(0x301)) == "Paraná")
    nfd = unicodedata.normalize("NFD", "Paraná")
    check("a precomposed form matches decomposed text",
          spelling.Spellings([{"from": ["Paraná"], "to": "P!"}]).apply(nfd + " x") == "P! x")
    check("a decomposed form matches precomposed text",
          spelling.Spellings([{"from": [nfd], "to": "P!"}]).apply("Paraná x") == "P! x")
    check("a decomposed rule still respects the word boundary",
          spelling.Spellings([{"from": [nfd], "to": "P!"}]).apply("Paranás") == "Paranás")
    check("multi-word 'to' goes in verbatim, never matched again",
          rules.apply("Rio SPPR") == "Rio de Janeiro Sao Paulo Parana", rules.apply("Rio SPPR"))
    accented = spelling.Spellings([{"from": ["paraná"], "to": "Paraná"}])
    check("accented 'from' matches regardless of case", accented.apply("PARANÁ, paraná") == "Paraná, Paraná",
          accented.apply("PARANÁ, paraná"))
    t = stt.Transcript(provider="x", text="sao paulo", utterances=[stt.Utterance("parana"), stt.Utterance("ok")])
    rules.apply_transcript(t)
    check("applies to the text and every utterance",
          t.text == "São Paulo" and [u.text for u in t.utterances] == ["Paraná", "ok"])
    for bad in ([{"from": [], "to": "x"}], [{"from": ["a"]}], {"from": ["a"], "to": "b"}, ["a"]):
        check(f"bad rules refused: {json.dumps(bad)}", isinstance(raised(lambda: spelling.Spellings(bad)), ValueError))


def test_soniox_tokens():
    section("soniox token grouping")
    t = to_transcript(SONIOX_TOKENS)
    check("speakers become A/B in order of appearance",
          [u.speaker for u in t.utterances] == ["A", "B", "A"], t.utterances)
    check("token text concatenates", t.utterances[0].text == "Bom dia, Ana.", t.utterances[0].text)
    check("utterance timestamps span its tokens",
          (t.utterances[1].start_ms, t.utterances[1].end_ms) == (1000, 1500))
    check("utterance language is its tokens' language", [u.language for u in t.utterances] == ["pt", "en", "pt"])
    check("format is Speaker blocks",
          t.format() == "Speaker A: Bom dia, Ana.\n\nSpeaker B: Hi there.\n\nSpeaker A: Tudo bem?", t.format())
    plain = to_transcript([{k: v for k, v in tok.items() if k != "speaker"} for tok in SONIOX_TOKENS])
    check("no speakers -> plain text", plain.format() == "Bom dia, Ana. Hi there. Tudo bem?", plain.format())


def test_soniox(work, audio):
    section("soniox provider")
    cfg = {"soniox_api_key": "sk-test", "keyterms": "Kubernetes, PostgreSQL, Kubernetes", "languages": "pt,en"}

    MOCK.reset()
    MOCK.polls_until_done = 2
    notes = []
    t = stt.transcribe(audio, config=cfg, progress=notes.append)
    check("returns the grouped transcript", t.provider == "soniox" and len(t.utterances) == 3)
    _, _, headers, upload = MOCK.requests[0]
    check("uploads the file as multipart to /v1/files",
          calls()[0] == ("POST", "/v1/files") and b"fake audio bytes" in upload
          and b'filename="clip.mp3"' in upload
          and headers.get("Content-Type", "").startswith("multipart/form-data"))
    check("bearer auth", headers.get("Authorization") == "Bearer sk-test")
    body = json_body("/v1/transcriptions")
    check("request: model, file, speakers, hints, terms (deduped)",
          body == {"model": "stt-async-v5", "file_id": "file-1", "enable_speaker_diarization": True,
                   "enable_language_identification": True, "language_hints": ["pt", "en"],
                   "context": {"terms": ["Kubernetes", "PostgreSQL"]}}, body)
    check("polls until completed", MOCK.polls == 3, MOCK.polls)
    check("deletes transcription and file afterwards", set(calls("DELETE")) == SONIOX_DELETES, calls("DELETE"))
    check("progress names the provider", "Uploading audio to Soniox..." in notes, notes)

    MOCK.reset()
    stt.transcribe(audio, config=cfg, speakers=False, terms=[], languages=[])
    body = json_body("/v1/transcriptions")
    check("overrides: no speakers, no terms, no hints",
          body["enable_speaker_diarization"] is False and "context" not in body and "language_hints" not in body, body)

    for state in ("error", "failed"):
        MOCK.reset()
        MOCK.fail_job = True
        MOCK.fail_state = state
        e = raised(lambda: stt.transcribe(audio, config=cfg))
        check(f"status {state!r} raises JobFailed with the provider's reason",
              isinstance(e, stt.JobFailed) and "audio decode failed" in str(e), repr(e))
        check(f"status {state!r}: still cleaned up", set(calls("DELETE")) == SONIOX_DELETES, calls("DELETE"))

    MOCK.reset()
    MOCK.create_status = 400
    MOCK.error_body = {"error_message": "bad request"}
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("rejected request raises with the body",
          isinstance(e, stt.rest.ApiError) and e.status == 400 and "bad request" in str(e), repr(e))
    check("rejected request deletes the upload, no retry",
          calls("DELETE") == [("DELETE", "/v1/files/file-1")] and len(calls("POST")) == 2, calls())

    MOCK.reset()
    MOCK.create_body = {"status": "queued"}
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("an answer without an id is an internal error that still deletes the upload",
          isinstance(e, KeyError) and calls("DELETE") == [("DELETE", "/v1/files/file-1")], (repr(e), calls()))

    MOCK.reset()
    MOCK.tokens = []
    t = stt.transcribe(audio, config=cfg)
    check("silence: empty text, no utterances, success", t.text == "" and t.utterances == [], t)


def test_blips_and_polling(audio):
    section("polling: fast start, blip window")
    cfg = {"soniox_api_key": "sk-test"}

    MOCK.reset()
    MOCK.poll_fail_count = 8
    t = stt.transcribe(audio, config=cfg)
    check("8 failed status checks inside the window are waited out (no count limit)",
          t.provider == "soniox" and MOCK.polls == 9, MOCK.polls)

    MOCK.reset()
    MOCK.poll_status = 503
    started = time.monotonic()
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    took = time.monotonic() - started
    check("persistent 5xx gives up after the window (time, not a count)",
          isinstance(e, stt.StillRunning) and 0.3 <= took < 2 and "status checks failed" in str(e), (took, repr(e)))
    check("... and the job is deleted by default, saying so",
          set(calls("DELETE")) == SONIOX_DELETES and "deleted" in str(e) and "may still finish" not in str(e), str(e))

    MOCK.reset()
    MOCK.poll_status = 401
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("a non-transient status error fails at once",
          isinstance(e, stt.rest.ApiError) and e.status == 401 and MOCK.polls == 1, (repr(e), MOCK.polls))

    saved = (Soniox.poll_seconds, stt.FAST_POLL_SECONDS, stt.FAST_POLL_WINDOW)
    Soniox.poll_seconds, stt.FAST_POLL_SECONDS, stt.FAST_POLL_WINDOW = 0.4, 0.04, 0.25
    try:
        MOCK.reset()
        MOCK.polls_until_done = 8
        stt.transcribe(audio, config=cfg)
        gaps = [b - a for a, b in zip(MOCK.poll_times, MOCK.poll_times[1:])]
        early = [g for g, at in zip(gaps, MOCK.poll_times[1:]) if at - MOCK.poll_times[0] < 0.2]
        check("first polls come fast, later ones at poll_seconds",
              len(early) >= 3 and max(early) < 0.15 and gaps[-1] >= 0.3, [round(g, 3) for g in gaps])
    finally:
        Soniox.poll_seconds, stt.FAST_POLL_SECONDS, stt.FAST_POLL_WINDOW = saved


def test_submit_retries(audio):
    section("submit retries")
    cfg = {"soniox_api_key": "sk-test"}

    MOCK.reset()
    MOCK.script = {("POST", "/v1/files"): [503, 429]}
    notes = []
    t = stt.transcribe(audio, config=cfg, progress=notes.append)
    check("a transient upload failure is retried twice",
          t.provider == "soniox" and calls("POST").count(("POST", "/v1/files")) == 3, calls())
    check("the retry is announced", any("retrying" in n for n in notes), notes)

    MOCK.reset()
    MOCK.script = {("POST", "/v1/files"): [503, 503, 503]}
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("three transient failures: the last error is raised",
          isinstance(e, stt.rest.ApiError) and e.status == 503
          and calls("POST").count(("POST", "/v1/files")) == 3, (repr(e), calls()))

    MOCK.reset()
    MOCK.script = {("POST", "/v1/transcriptions"): [500]}
    t = stt.transcribe(audio, config=cfg)
    check("a transient create failure: the upload is deleted, the whole submit retried",
          t.provider == "soniox" and calls("POST").count(("POST", "/v1/files")) == 2
          and calls("DELETE")[0] == ("DELETE", "/v1/files/file-1"), calls())

    MOCK.reset()
    MOCK.script = {("POST", "/v1/files"): [402]}
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("a 4xx is not retried", isinstance(e, stt.rest.ApiError) and len(calls("POST")) == 1, calls())

    e = raised(lambda: stt.transcribe("/no/such/file.mp3", config=cfg))
    check("a missing file is not retried (FileNotFoundError)", isinstance(e, FileNotFoundError), repr(e))

    # The provider took the create request and the answer never came: it may
    # have made (and will bill) a job stt never learns the id of, so trying
    # again could make a second one.
    for name, conf, create in (("assemblyai", {"provider": "assemblyai", "assemblyai_api_key": "aai-test"},
                                "/v2/transcript"),
                               ("soniox", cfg, "/v1/transcriptions")):
        MOCK.reset()
        MOCK.drop = {("POST", create): 3}
        e = raised(lambda: stt.transcribe(audio, config=conf))
        check(f"{name}: a lost answer to creating the job is not retried, and the message says why",
              isinstance(e, ConnectionError) and stt.rest.is_transient(e)
              and calls("POST").count(("POST", create)) == 1 and "may exist that stt cannot see" in str(e),
              (repr(e), calls()))
    check("soniox: ... its file is deleted, so such a transcription fails",
          calls("DELETE") == [("DELETE", "/v1/files/file-1")] and "its file was deleted" in str(e), (calls(), str(e)))

    MOCK.reset()
    MOCK.drop = {("POST", "/v1/files"): 1}
    t = stt.transcribe(audio, config=cfg)
    check("a lost answer to an upload is retried (no job exists yet)",
          t.provider == "soniox" and calls("POST").count(("POST", "/v1/files")) == 2, calls())

    MOCK.reset()
    MOCK.create_status = 503
    MOCK.delete_status = {"/v1/files/file-1": 500}
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("a transient create failure whose upload cannot be deleted: named, not retried",
          isinstance(e, ConnectionError) and "Soniox kept the file" in str(e)
          and calls("POST").count(("POST", "/v1/files")) == 1, (repr(e), calls()))

    MOCK.reset()
    MOCK.create_status = 400
    MOCK.delete_status = {"/v1/files/file-1": 500}
    notes = []
    e = raised(lambda: stt.transcribe(audio, config=cfg, progress=notes.append))
    check("a refused create whose upload cannot be deleted says so",
          isinstance(e, stt.rest.ApiError) and e.status == 400
          and any("Could not delete the Soniox upload file-1" in n for n in notes), notes)


def test_answer_lost(base):
    section("network errors after the request went out")
    import errno
    import http.client
    import ssl
    import urllib.error
    original = http.client.HTTPConnection.getresponse
    for error in (ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
                  OSError(errno.EHOSTUNREACH, "No route to host"), TimeoutError("timed out")):
        def broken(self, *args, **kwargs):
            raise error
        http.client.HTTPConnection.getresponse = broken
        try:
            e = raised(lambda: stt.rest.request("GET", base + "/v1/transcriptions/tr-1"))
        finally:
            http.client.HTTPConnection.getresponse = original
        check(f"{type(error).__name__} while waiting for the answer: AnswerLost, a transient network error",
              isinstance(e, stt.rest.AnswerLost) and stt.rest.is_transient(e), repr(e))
    e = raised(lambda: stt.rest.request("GET", f"http://127.0.0.1:{free_port()}/x"))
    check("connection refused stays URLError (nothing was sent)",
          isinstance(e, urllib.error.URLError) and not isinstance(e, stt.rest.AnswerLost), repr(e))


def test_timeouts(audio):
    section("timeouts and retention")
    cfg = {"soniox_api_key": "sk-test"}

    MOCK.reset()
    MOCK.polls_until_done = 10 ** 6
    e = raised(lambda: stt.transcribe(audio, config=cfg, timeout=0.05))
    check("timeout raises StillRunning", isinstance(e, stt.StillRunning) and "timed out" in str(e), repr(e))
    check("timeout deletes the job by default and says so",
          set(calls("DELETE")) == SONIOX_DELETES and "was deleted" in str(e), str(e))

    MOCK.reset()
    MOCK.polls_until_done = 10 ** 6
    e = raised(lambda: stt.transcribe(audio, config=cfg, timeout=0.05, keep_on_timeout=True))
    check("keep_on_timeout leaves the job and names it",
          isinstance(e, stt.StillRunning) and calls("DELETE") == [] and "tr-1 may still finish" in str(e), str(e))

    MOCK.reset()
    MOCK.polls_until_done = 10 ** 6
    MOCK.delete_status = {"/v1/transcriptions/tr-1": 409}
    MOCK.error_body = {"error_type": "transcription_invalid_state", "message": "still processing"}
    e = raised(lambda: stt.transcribe(audio, config=cfg, timeout=0.05))
    check("a refused delete is reported, not looped on",
          isinstance(e, stt.StillRunning) and "may still finish" in str(e) and "still running" in str(e)
          and calls("DELETE").count(("DELETE", "/v1/transcriptions/tr-1")) == 1, (str(e), calls("DELETE")))
    check("... the file is deleted anyway", ("DELETE", "/v1/files/file-1") in calls("DELETE"))
    check("... and the message stays one short line", len(str(e)) <= 320 and "\n" not in str(e), len(str(e)))


def test_deliver(audio):
    section("deliver before cleanup")
    cfg = {"soniox_api_key": "sk-test"}
    MOCK.reset()
    seen = []
    stt.transcribe(audio, config=cfg, deliver=lambda t: seen.append((t.text, list(calls("DELETE")))))
    check("deliver gets the transcript before any DELETE",
          len(seen) == 1 and seen[0][0].startswith("Bom dia") and seen[0][1] == [], seen)
    check("... and the job is deleted after", set(calls("DELETE")) == SONIOX_DELETES)

    MOCK.reset()

    def broken(_t):
        raise OSError("disk full")
    e = raised(lambda: stt.transcribe(audio, config=cfg, deliver=broken))
    check("a failing deliver still cleans up", isinstance(e, OSError) and set(calls("DELETE")) == SONIOX_DELETES)

    MOCK.reset()
    rules = [{"from": ["ana"], "to": "ANA"}]
    t = stt.transcribe(audio, config=cfg, spellings=rules,
                       deliver=lambda t: seen.append(t.utterances[0].text))
    check("spellings apply before deliver, to text and utterances",
          "ANA" in t.text and seen[-1] == "Bom dia, ANA.", (t.text, seen[-1]))

    MOCK.reset()
    e = raised(lambda: stt.transcribe(audio, config=cfg, spellings=[{"from": "x"}]))
    check("bad spellings fail before any upload", isinstance(e, ValueError) and MOCK.requests == [], repr(e))


def test_errors(audio):
    section("error text")
    key = "sk-secret-0123456789"
    cfg = {"soniox_api_key": key}
    MOCK.reset()
    MOCK.create_status = 400
    MOCK.error_body = {"message": f"key {key} refused " + "x" * 2000}
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("the provider's body is cut to ~300 characters",
          isinstance(e, stt.rest.ApiError) and len(str(e)) <= 340, len(str(e)))
    check("the API key is scrubbed", key not in str(e) and "key *** refused" in str(e), str(e)[:80])

    MOCK.reset()
    MOCK.create_status = 502
    MOCK.error_body = b"<html>\n  upstream " + key.encode() + b" down\n" + b"y" * 1000
    MOCK.script = {}
    Soniox.submit_retry_waits = ()
    try:
        e = raised(lambda: stt.transcribe(audio, config=cfg))
    finally:
        Soniox.submit_retry_waits = (0.01, 0.02)
    check("a non-JSON body: one line, cut, scrubbed",
          isinstance(e, stt.rest.ApiError) and "\n" not in str(e) and len(str(e)) <= 340
          and key not in str(e) and "upstream *** down" in str(e), str(e)[:80])

    aai_key = "aai-secret-987654"
    MOCK.reset()
    MOCK.create_status = 400
    MOCK.error_body = {"error": f"Authorization {aai_key} invalid"}
    e = raised(lambda: stt.transcribe(audio, config={"provider": "assemblyai", "assemblyai_api_key": aai_key}))
    check("AssemblyAI's raw key is scrubbed too", isinstance(e, stt.rest.ApiError) and aai_key not in str(e), str(e))

    # Where a key could survive: echoed in the reason phrase (retry lines
    # print it), or cut in half by a clip that ran before the scrub.
    MOCK.reset()
    MOCK.script = {("POST", "/v1/files"): [503, 503, 503]}
    MOCK.reason = f"Bad {key}"
    notes = []
    e = raised(lambda: stt.transcribe(audio, config=cfg, progress=notes.append))
    check("a key in the HTTP reason phrase is scrubbed, retry lines included",
          isinstance(e, stt.rest.ApiError) and any("retrying" in n for n in notes)
          and key[:6] not in str(e) + " ".join(notes), notes)

    MOCK.reset()
    MOCK.create_body = b"<html>" + b" " * 180 + key.encode() + b"</html>"
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("a key across the cut of a non-JSON answer leaves no prefix",
          isinstance(e, ValueError) and "not JSON" in str(e) and key[:6] not in str(e), str(e))

    for name, conf, secret in (("soniox", cfg, key),
                               ("assemblyai", {"provider": "assemblyai", "assemblyai_api_key": aai_key}, aai_key)):
        MOCK.reset()
        MOCK.fail_job = True
        MOCK.job_error = "z" * 290 + secret
        e = raised(lambda: stt.transcribe(audio, config=conf))
        check(f"{name}: a key at the cut of a failed job's error text is scrubbed first",
              isinstance(e, stt.JobFailed) and secret[:6] not in str(e) and "***" in str(e), str(e)[-40:])


def test_assemblyai(audio):
    section("assemblyai provider")
    cfg = {"provider": "assemblyai", "assemblyai_api_key": "aai-test", "keyterms": "Kubernetes", "languages": "pt,en"}

    MOCK.reset()
    t = stt.transcribe(audio, config=cfg)
    _, _, headers, upload = MOCK.requests[0]
    check("uploads raw bytes to /v2/upload",
          calls()[0] == ("POST", "/v2/upload") and upload == b"fake audio bytes")
    check("raw key as Authorization", headers.get("Authorization") == "aai-test")
    body = json_body("/v2/transcript")
    check("request: models, speakers, detection over the hints with the first as fallback, terms",
          body == {"audio_url": "https://cdn.example/audio", "speech_models": ["universal-3-5-pro", "universal-2"],
                   "speaker_labels": True, "language_detection": True,
                   "language_detection_options": {"expected_languages": ["pt", "en"], "fallback_language": "pt",
                                                  "on_no_speech_detected": "fallback"},
                   "keyterms_prompt": ["Kubernetes"]}, body)
    check("utterances map through",
          t.format() == "Speaker A: Bom dia.\n\nSpeaker B: Hi there." and t.utterances[1].start_ms == 1000, t.format())
    check("each utterance carries language_code", [u.language for u in t.utterances] == ["pt", "pt"], t.utterances)
    check("the transcript is deleted afterwards", calls("DELETE") == [("DELETE", "/v2/transcript/aai-1")], calls("DELETE"))

    MOCK.reset()
    stt.transcribe(audio, config=cfg, languages=[], terms=[])
    body = json_body("/v2/transcript")
    check("no hints: detection over every language, no options, no terms",
          body["language_detection"] is True and "language_detection_options" not in body
          and "keyterms_prompt" not in body, body)

    MOCK.reset()
    many = [f"term {i}" for i in range(1100)]
    terms = ["Kubernetes", "Kubernetes", "one two three four five six", "one two three four five six seven"] + many
    stt.transcribe(audio, config=cfg, terms=terms)
    sent = json_body("/v2/transcript")["keyterms_prompt"]
    check("key terms: deduped, none over six words, first 1000",
          len(sent) == 1000 and sent[:3] == ["Kubernetes", "one two three four five six", "term 0"]
          and "one two three four five six seven" not in sent, (len(sent), sent[:4]))

    MOCK.reset()
    MOCK.aai_result = {**AAI_DONE, "utterances": None}
    t = stt.transcribe(audio, config=cfg, speakers=False)
    check("no speaker labels: one utterance with the text, times and language",
          len(t.utterances) == 1 and t.utterances[0].text == "Bom dia. Hi there."
          and (t.utterances[0].start_ms, t.utterances[0].end_ms, t.utterances[0].language) == (0, 1500, "pt"),
          t.utterances)

    MOCK.reset()
    MOCK.aai_result = {"id": "aai-1", "status": "completed", "text": None, "utterances": [], "language_code": "en"}
    t = stt.transcribe(audio, config=cfg)
    check("silence: completed with no text = success, empty", t.text == "" and t.utterances == [], t)

    MOCK.reset()
    MOCK.fail_job = True
    e = raised(lambda: stt.transcribe(audio, config=cfg))
    check("failed job raises JobFailed", isinstance(e, stt.JobFailed) and "audio decode failed" in str(e))
    check("failed job is deleted too", calls("DELETE") == [("DELETE", "/v2/transcript/aai-1")])

    MOCK.reset()
    MOCK.polls_until_done = 10 ** 6
    MOCK.delete_status = {"/v2/transcript/aai-1": 400}
    MOCK.error_body = {"error": "Transcript is still processing"}
    e = raised(lambda: stt.transcribe(audio, config=cfg, timeout=0.05))
    check("timeout: AssemblyAI refusing to delete a running transcript is reported",
          isinstance(e, stt.StillRunning) and "aai-1 may still finish" in str(e) and "AssemblyAI kept" in str(e),
          str(e))


def test_duration_guard(work):
    section("duration guard")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        check("ffmpeg/ffprobe present for the duration guard", False, "install ffmpeg")
        return None
    tone = os.path.join(work, "tone.mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-ac", "1", tone], check=True)
    cfg = {"soniox_api_key": "sk-test"}
    saved = Soniox.max_seconds
    Soniox.max_seconds = 1
    try:
        MOCK.reset()
        e = raised(lambda: stt.transcribe(tone, config=cfg))
        check("longer than max_seconds: TooLong naming both numbers, nothing uploaded",
              isinstance(e, stt.TooLong) and "2 s" in str(e) and "at most 1 s" in str(e) and MOCK.requests == [],
              (repr(e), calls()))
        check("TooLong is a JobFailed (not worth retrying)", isinstance(e, stt.JobFailed))

        MOCK.reset()
        path = os.environ["PATH"]
        os.environ["PATH"] = os.path.join(work, "no-bin")
        try:
            t = stt.transcribe(tone, config=cfg)
        finally:
            os.environ["PATH"] = path
        check("without ffprobe the guard is skipped", t.provider == "soniox" and ("POST", "/v1/files") in calls())

        MOCK.reset()
        t = stt.transcribe(open(tone, "rb").read(), config=cfg)
        check("bytes input skips the guard", t.provider == "soniox" and ("POST", "/v1/files") in calls())
    finally:
        Soniox.max_seconds = saved

    MOCK.reset()
    t = stt.transcribe(tone, config=cfg)
    check("under max_seconds: uploaded", t.provider == "soniox" and ("POST", "/v1/files") in calls())
    return tone


def test_fake(audio):
    section("fake provider")
    cfg = {"provider": "fake"}
    check("needs no key", stt.config_problem(cfg) is None)
    t = stt.transcribe(audio, config=cfg)
    check("deterministic transcript naming file and size",
          t.provider == "fake" and t.text == "fake transcript of clip.mp3 (16 bytes)"
          and [(u.text, u.speaker, u.language, u.start_ms) for u in t.utterances]
          == [(t.text, "A", "en", 0)], t)
    t = stt.transcribe(b"12345", config=cfg, speakers=False)
    check("bytes: named 'audio', no speaker without speakers",
          t.text == "fake transcript of audio (5 bytes)" and t.utterances[0].speaker is None, t)
    check("registered", "fake" in stt.providers.names())

    expect = {"auth": stt.rest.ApiError, "server": stt.rest.ApiError, "network": ConnectionError,
              "config": stt.ConfigError, "limit": stt.TooLong, "internal": ValueError}
    for kind, cls in expect.items():
        e = raised(lambda: stt.transcribe(audio, config={"provider": "fake", "fake_fail": f"submit:{kind}"}))
        check(f"fake_fail=submit:{kind} raises {cls.__name__}", isinstance(e, cls), repr(e))
    e = raised(lambda: stt.transcribe(audio, config={"provider": "fake", "fake_fail": "job"}))
    check("fake_fail=job raises JobFailed", isinstance(e, stt.JobFailed), repr(e))
    started = time.monotonic()
    t = stt.transcribe(audio, config={"provider": "fake", "fake_fail": "slow:0.4"})
    check("fake_fail=slow:S takes about S", t.provider == "fake" and time.monotonic() - started >= 0.4)
    e = raised(lambda: stt.transcribe(audio, config={"provider": "fake", "fake_fail": "slow:5"}, timeout=0.2))
    check("slow past the timeout: StillRunning", isinstance(e, stt.StillRunning) and "deleted" in str(e), repr(e))
    check("stream-only knobs are ignored", stt.config_problem({"provider": "fake", "fake_fail": "connect:auth"}) is None)
    check("unknown knob = config problem",
          "fake_fail" in (stt.config_problem({"provider": "fake", "fake_fail": "submit:nope"}) or ""))


def test_cli(work, audio, base, tone):
    section("stt command")
    cfg_path = os.path.join(work, "cli-config")
    write_config(cfg_path, provider="soniox", soniox_api_key="sk-cli", keyterms="Kubernetes")
    cli = Cli(base, cfg_path)

    MOCK.reset()
    r = cli.run("transcribe", audio)
    check("transcribe: exit 0, Speaker text on stdout",
          r.returncode == 0 and r.stdout.startswith("Speaker A: Bom dia, Ana."), (r.returncode, r.stdout, r.stderr))
    check("progress goes to stderr", "Uploading audio to Soniox" in r.stderr and "Uploading" not in r.stdout)

    MOCK.reset()
    r = cli.run("transcribe", "--json", audio)
    data = json.loads(r.stdout) if r.returncode == 0 else {}
    check("--json carries utterances with timestamps and language",
          data.get("provider") == "soniox" and data["utterances"][1]["start_ms"] == 1000
          and data["utterances"][1]["language"] == "en", r.stdout[:200])

    out = os.path.join(work, "out.txt")
    MOCK.reset()
    r = cli.run("transcribe", "-o", out, audio)
    check("-o writes the file, nothing on stdout",
          r.returncode == 0 and r.stdout == "" and open(out).read().startswith("Speaker A:"), r.stderr)

    # -o that cannot be written: refused before the (paid) transcription
    outdir = os.path.join(work, "outdir")
    os.mkdir(outdir)
    MOCK.reset()
    r = cli.run("transcribe", "-o", outdir, audio)
    check("-o naming a directory: exit 2 before any upload",
          failed_cleanly(r, 2, "is a directory") and MOCK.requests == [], (r.returncode, r.stderr))
    if os.geteuid() != 0:
        readonly = os.path.join(work, "readonly")
        os.mkdir(readonly)
        os.chmod(readonly, 0o500)
        try:
            MOCK.reset()
            r = cli.run("transcribe", "-o", os.path.join(readonly, "out.txt"), audio)
        finally:
            os.chmod(readonly, 0o700)
        check("-o in a read-only directory: exit 2 before any upload",
              failed_cleanly(r, 2, "cannot create files") and MOCK.requests == [], (r.returncode, r.stderr))

    # stdout that cannot take the transcript: one defined exit, the job still deleted
    MOCK.reset()
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        r = cli.run("transcribe", audio, stdout=write_end)
    finally:
        os.close(write_end)
    check("closed stdout: exit 1, 'stt:' line last, no shutdown traceback",
          r.returncode == 1 and last_line(r.stderr).startswith("stt: cannot write to stdout")
          and "Exception ignored" not in r.stderr, (r.returncode, r.stderr[-300:]))
    check("closed stdout: the job is deleted anyway", set(calls("DELETE")) == SONIOX_DELETES, calls("DELETE"))
    if os.path.exists("/dev/full"):
        MOCK.reset()
        with open("/dev/full", "w") as full:
            r = cli.run("transcribe", "--json", audio, stdout=full)
        check("full disk on stdout: exit 1, 'stt:' line last, no shutdown traceback",
              r.returncode == 1 and "No space left" in last_line(r.stderr)
              and last_line(r.stderr).startswith("stt: ") and "Exception ignored" not in r.stderr,
              (r.returncode, r.stderr[-300:]))

    # a pipe as FILE (a FIFO, or bash's <(...)): read once, ffprobe never sees it
    fifo = os.path.join(work, "audio.fifo")
    os.mkfifo(fifo)

    def feed():
        with open(fifo, "wb") as f:
            f.write(b"fifo audio bytes")
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    MOCK.reset()
    r = cli.run("transcribe", fifo)
    if feeder.is_alive():  # the command never opened it: let the writer go
        os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))
    upload = MOCK.requests[0][3] if MOCK.requests else b""
    check("a named pipe as FILE: exit 0, the whole audio uploaded",
          r.returncode == 0 and b"fifo audio bytes" in upload, (r.returncode, r.stderr[-300:]))
    r = cli.run("transcribe", work)
    check("a directory as FILE: exit 2", failed_cleanly(r, 2, "is a directory"), r.stderr)

    MOCK.reset()
    MOCK.tokens = []
    r = cli.run("transcribe", "--json", audio)
    check("silence: exit 0 with empty text",
          r.returncode == 0 and json.loads(r.stdout)["text"] == "", (r.returncode, r.stdout, r.stderr))

    # stdin, without a temp file
    tmp = os.path.join(work, "tmpdir")
    os.makedirs(tmp)
    MOCK.reset()
    r = cli.run("transcribe", "-", stdin=b"fake audio bytes", text=False, env={"TMPDIR": tmp})
    upload = MOCK.requests[0][3] if MOCK.requests else b""
    check("stdin: exit 0, uploaded from memory as 'audio'",
          r.returncode == 0 and b"fake audio bytes" in upload and b'filename="audio"' in upload,
          (r.returncode, r.stderr[-300:]))
    check("stdin: no file appeared in TMPDIR", os.listdir(tmp) == [], os.listdir(tmp))
    r = cli.run("transcribe", "-", stdin=b"", text=False)
    check("stdin empty: exit 2", r.returncode == 2 and last_line(r.stderr.decode()).startswith("stt: "))

    # --terms-file
    terms_file = os.path.join(work, "terms.txt")
    with open(terms_file, "w") as f:
        f.write("Earth, Wind & Fire\n# a comment\n\nKubernetes\nKubernetes\n")
    MOCK.reset()
    r = cli.run("transcribe", "--terms-file", terms_file, audio)
    body = json_body("/v1/transcriptions") or {}
    check("--terms-file: one term per line, commas kept",
          r.returncode == 0 and body.get("context") == {"terms": ["Earth, Wind & Fire", "Kubernetes"]}, body)
    r = cli.run("transcribe", "--terms", "a", "--terms-file", terms_file, audio)
    check("--terms and --terms-file together: exit 2, stt: line", failed_cleanly(r, 2, "not allowed"), r.stderr)
    MOCK.reset()
    r = cli.run("transcribe", "--terms", "", audio)
    check("--terms '' sends none", r.returncode == 0 and "context" not in (json_body("/v1/transcriptions") or {}))

    # --spellings
    rules_file = os.path.join(work, "spellings.json")
    with open(rules_file, "w") as f:
        json.dump([{"from": ["Ana"], "to": "Ana Lúcia"}], f)
    MOCK.reset()
    r = cli.run("transcribe", "--spellings", rules_file, "--json", audio)
    data = json.loads(r.stdout) if r.returncode == 0 else {}
    check("--spellings applies to text and utterances",
          "Ana Lúcia." in data.get("text", "") and data["utterances"][0]["text"] == "Bom dia, Ana Lúcia.",
          r.stdout[:200])
    with open(rules_file, "w") as f:
        f.write("[{")
    MOCK.reset()
    r = cli.run("transcribe", "--spellings", rules_file, audio)
    check("--spellings broken: exit 2 before any upload", failed_cleanly(r, 2, "--spellings") and MOCK.requests == [],
          r.stderr)

    # exit statuses
    r = cli.run("transcribe", os.path.join(work, "missing.mp3"))
    check("missing file: exit 2", failed_cleanly(r, 2, "no such file"), r.stderr)

    empty = os.path.join(work, "no-such-config")
    r = cli.run("transcribe", audio, env={"STT_CONFIG": empty})
    check("no key: exit 3", failed_cleanly(r, 3, "soniox_api_key is not set"), (r.returncode, r.stderr))

    MOCK.reset()
    MOCK.fail_job = True
    r = cli.run("transcribe", audio)
    check("failed job: exit 4, reason on stderr, nothing on stdout",
          failed_cleanly(r, 4, "audio decode failed"), (r.returncode, r.stderr))

    MOCK.reset()
    MOCK.create_status = 400
    MOCK.error_body = {"message": "key sk-cli rejected " + "z" * 1000}
    r = cli.run("transcribe", audio)
    check("rejected request: exit 4, line cut and scrubbed",
          failed_cleanly(r, 4, "HTTP 400") and "sk-cli" not in r.stderr + r.stdout, r.stderr[-400:])

    MOCK.reset()
    MOCK.create_status = 503
    r = cli.run("transcribe", audio)
    check("5xx after the retries: exit 5", failed_cleanly(r, 5, "HTTP 503")
          and calls("POST").count(("POST", "/v1/transcriptions")) == 3, (r.returncode, r.stderr, calls()))

    port = free_port()
    r = cli.run("transcribe", audio, setup=f"Soniox.base_url = 'http://127.0.0.1:{port}'")
    check("network down: exit 5", failed_cleanly(r, 5), (r.returncode, r.stderr[-300:]))

    MOCK.reset()
    r = cli.run("transcribe", audio, setup=(
        "import errno, http.client\n"
        "def unreachable(self, *args, **kwargs):\n"
        "    raise OSError(errno.EHOSTUNREACH, 'No route to host')\n"
        "http.client.HTTPConnection.getresponse = unreachable\n"))
    check("no route while waiting for an answer: exit 5, not an internal error",
          failed_cleanly(r, 5, "no answer came") and "internal error" not in r.stderr, (r.returncode, r.stderr[-300:]))

    MOCK.reset()
    MOCK.drop = {("POST", "/v1/transcriptions"): 3}
    r = cli.run("transcribe", audio)
    check("lost answer to the create: exit 5, said once, not retried",
          failed_cleanly(r, 5, "may exist that stt cannot see")
          and calls("POST").count(("POST", "/v1/transcriptions")) == 1, (r.returncode, r.stderr[-300:]))

    MOCK.reset()
    MOCK.create_status = 503
    MOCK.error_body = b"<html>" + b"upstream unavailable " * 40 + b"</html>"
    MOCK.delete_status = {"/v1/files/file-1": 500}
    r = cli.run("transcribe", audio)
    check("an undeletable upload behind a long HTML error: 'not retried' survives the clip",
          failed_cleanly(r, 5, "not retried") and "Soniox kept the file" in last_line(r.stderr),
          (r.returncode, r.stderr[-300:]))

    MOCK.reset()
    MOCK.script = {("POST", "/v1/files"): [503, 503, 503]}
    MOCK.reason = "Bad sk-cli"
    r = cli.run("transcribe", audio)
    check("a key echoed in the reason phrase never reaches stderr, retry lines included",
          failed_cleanly(r, 5, "HTTP 503") and "retrying" in r.stderr and "sk-cli" not in r.stderr,
          r.stderr[-400:])

    MOCK.reset()
    MOCK.polls_until_done = 10 ** 6
    r = cli.run("transcribe", "--timeout", "0.2", audio)
    check("timeout: exit 6, job deleted, message says so",
          failed_cleanly(r, 6, "was deleted") and set(calls("DELETE")) == SONIOX_DELETES, (r.returncode, r.stderr))

    MOCK.reset()
    MOCK.polls_until_done = 10 ** 6
    r = cli.run("transcribe", "--timeout", "0.2", "--keep-on-timeout", audio)
    check("--keep-on-timeout: exit 6, nothing deleted, job id named",
          failed_cleanly(r, 6, "tr-1 may still finish") and calls("DELETE") == [], (r.returncode, r.stderr))

    MOCK.reset()
    MOCK.poll_status = 503
    r = cli.run("transcribe", audio, setup="stt.POLL_ERROR_WINDOW = 0.3")
    check("status checks failing past the window: exit 6", failed_cleanly(r, 6, "status checks failed"),
          (r.returncode, r.stderr[-300:]))

    MOCK.reset()
    MOCK.create_body = {"status": "queued"}
    r = cli.run("transcribe", audio)
    check("unexpected answer: exit 1, one 'internal error' line, no traceback",
          failed_cleanly(r, 1, "stt: internal error: KeyError") and "Traceback" not in r.stderr, r.stderr)
    MOCK.reset()
    MOCK.create_body = {"status": "queued"}
    r = cli.run("transcribe", audio, env={"STT_DEBUG": "1"})
    check("STT_DEBUG=1 adds the traceback, last line unchanged",
          "Traceback" in r.stderr and last_line(r.stderr).startswith("stt: internal error"), r.stderr[-300:])

    if tone:
        MOCK.reset()
        r = cli.run("transcribe", tone, setup="Soniox.max_seconds = 1")
        check("too long: exit 4 naming both numbers, nothing uploaded",
              failed_cleanly(r, 4, "at most 1 s") and "2 s" in r.stderr and MOCK.requests == [], r.stderr)

    # check
    r = cli.run("check")
    lines = r.stdout.splitlines()
    check("check: first line names provider and config",
          r.returncode == 0 and lines[0] == f"provider: soniox (config: {cfg_path})", r.stdout)
    check("check: second line is the stream state",
          len(lines) == 2 and lines[1].startswith("stream: "), r.stdout)
    r = cli.run("check", "--json")
    data = json.loads(r.stdout) if r.returncode == 0 else {}
    check("check --json: provider, config, ok, problem, max_seconds, stream",
          data.get("provider") == "soniox" and data.get("config") == cfg_path and data.get("ok") is True
          and data.get("problem") is None and data.get("max_seconds") == 5 * 3600
          and set(data.get("stream", {})) == {"ok", "problem", "model"}, r.stdout)
    r = cli.run("check", "--json", setup="sys.modules['stt.stream'] = None")
    data = json.loads(r.stdout) if r.returncode == 0 else {}
    check("check without stt.stream: stream not available",
          data.get("stream") == {"ok": False, "problem": "not available", "model": None}, r.stdout)
    r = cli.run("check", setup="sys.modules['stt.stream'] = None")
    check("check plain without stt.stream", r.stdout.splitlines()[1:] == ["stream: not available"], r.stdout)

    r = cli.run("check", env={"STT_CONFIG": empty})
    check("check without key: exit 3, first line kept, the reason last on stderr",
          r.returncode == 3 and r.stdout.startswith(f"provider: soniox (config: {empty})")
          and last_line(r.stderr) == f"stt: soniox_api_key is not set in {empty}", (r.returncode, r.stdout, r.stderr))
    r = cli.run("check", "--json", env={"STT_CONFIG": empty})
    data = json.loads(r.stdout) if r.stdout.strip() else {}
    check("check --json without key: ok false, problem, max_seconds, exit 3",
          r.returncode == 3 and data.get("ok") is False and "soniox_api_key" in (data.get("problem") or "")
          and data.get("max_seconds") == 5 * 3600, (r.returncode, r.stdout))
    r = cli.run("check", "--json", "--provider", "fake", env={"STT_CONFIG": empty})
    data = json.loads(r.stdout) if r.stdout.strip() else {}
    check("check --provider fake: ok without any key", r.returncode == 0 and data.get("provider") == "fake", r.stdout)


def test_cli_signals(work, audio, base):
    section("stt command: signals and ordering")
    cfg_path = os.path.join(work, "cli-config")
    cli = Cli(base, cfg_path)

    for sig, status in ((signal.SIGTERM, 143), (signal.SIGINT, 130)):
        MOCK.reset()
        MOCK.polls_until_done = 10 ** 6
        proc = cli.popen("transcribe", audio)
        holding = wait_for(lambda: MOCK.polls >= 2)
        started = time.monotonic()
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=30)
        check(f"{sig.name} while the provider holds the job: exit {status}, job deleted",
              holding and proc.returncode == status and set(calls("DELETE")) == SONIOX_DELETES and out == "",
              (proc.returncode, calls("DELETE"), err[-300:]))
        check(f"{sig.name}: quick", time.monotonic() - started < 5)
        check(f"{sig.name}: the last stderr line says so", last_line(err) == f"stt: cancelled by {sig.name}",
              err[-300:])

    # The first signal lands inside a cleanup that a failure started: the
    # interrupted DELETE is sent again and the file goes too.
    for label, arm, extra in (("a failed job", lambda: setattr(MOCK, "fail_job", True), ()),
                              ("a timeout", lambda: setattr(MOCK, "polls_until_done", 10 ** 6),
                               ("--timeout", "0.3"))):
        MOCK.reset()
        arm()
        MOCK.delete_gate = threading.Event()
        proc = cli.popen("transcribe", *extra, audio)
        reached = wait_for(lambda: len(calls("DELETE")) >= 1, 15)
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.2)
        MOCK.delete_gate.set()
        _, err = proc.communicate(timeout=30)
        check(f"SIGTERM inside the cleanup after {label}: everything still deleted, exit 143",
              reached and proc.returncode == 143 and set(calls("DELETE")) == SONIOX_DELETES,
              (proc.returncode, calls("DELETE"), err[-300:]))
        check(f"... after {label}: the last stderr line says cancelled",
              last_line(err) == "stt: cancelled by SIGTERM", err[-300:])

    # The transcript reaches stdout while the DELETEs are still held back.
    MOCK.reset()
    MOCK.delete_gate = threading.Event()
    proc = cli.popen("transcribe", "--json", audio)
    line = []
    reader = threading.Thread(target=lambda: line.append(proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(15)
    got = bool(line) and line[0].startswith("{")
    check("stdout has the transcript before any DELETE was answered",
          got and MOCK.deletes_answered == 0, (line, MOCK.deletes_answered))
    wait_for(lambda: len(calls("DELETE")) >= 1)
    proc.send_signal(signal.SIGTERM)  # mid-cleanup: must not cut it short
    time.sleep(0.2)
    MOCK.delete_gate.set()
    _, err = proc.communicate(timeout=30)
    check("SIGTERM during cleanup: cleanup finishes, then exit 143",
          proc.returncode == 143 and MOCK.deletes_answered == 2 and set(calls("DELETE")) == SONIOX_DELETES,
          (proc.returncode, MOCK.deletes_answered, err[-300:]))
    check("... with an 'stt:' line saying the transcript was already out",
          last_line(err) == "stt: SIGTERM arrived after the transcript was written", err[-300:])


def test_cli_fake(work, audio):
    section("stt command with the fake provider")
    cfg_path = os.path.join(work, "fake-config")
    cli = Cli("http://127.0.0.1:9", cfg_path)

    def run(*args, knob=None):
        write_config(cfg_path, provider="fake", **({"fake_fail": knob} if knob else {}))
        return cli.run(*args)

    r = run("transcribe", "--json", audio)
    data = json.loads(r.stdout) if r.returncode == 0 else {}
    check("fake: exit 0, deterministic JSON",
          data.get("text") == "fake transcript of clip.mp3 (16 bytes)"
          and data["utterances"][0]["speaker"] == "A", (r.returncode, r.stdout, r.stderr))
    for knob, status in (("submit:auth", 4), ("submit:credit", 4), ("submit:rejected", 4), ("submit:limit", 4),
                         ("submit:server", 5), ("submit:rate_limit", 5), ("submit:network", 5),
                         ("submit:timeout", 5), ("submit:config", 3), ("submit:internal", 1), ("job", 4)):
        r = run("transcribe", audio, knob=knob)
        check(f"fake_fail={knob}: exit {status}", failed_cleanly(r, status), (r.returncode, r.stderr[-300:]))
    r = run("transcribe", "--timeout", "0.3", audio, knob="slow:5")
    check("fake_fail=slow:5 with --timeout 0.3: exit 6", failed_cleanly(r, 6, "timed out"), r.stderr[-300:])
    r = run("check", knob="submit:nope")
    check("bad knob: check exit 3", r.returncode == 3 and "fake_fail" in last_line(r.stderr), r.stderr)
    # One fake_fail key for both fakes: each ignores the other's values.
    for knob in ("slow:5", "job", "after:1:network", "connect:auth"):
        r = run("check", knob=knob)
        check(f"fake_fail={knob}: check exit 0, both fakes ready",
              r.returncode == 0 and r.stdout.splitlines()[1:] == ["stream: ready (fake)"], (r.returncode, r.stdout))


def main():
    work = tempfile.mkdtemp(prefix="stt-test-")
    audio = os.path.join(work, "clip.mp3")
    with open(audio, "wb") as f:
        f.write(b"fake audio bytes")
    server, base = start_mock()
    try:
        test_config(work)
        test_spelling()
        test_soniox_tokens()
        test_soniox(work, audio)
        test_blips_and_polling(audio)
        test_submit_retries(audio)
        test_answer_lost(base)
        test_timeouts(audio)
        test_deliver(audio)
        test_errors(audio)
        test_assemblyai(audio)
        tone = test_duration_guard(work)
        test_fake(audio)
        test_cli(work, audio, base, tone)
        test_cli_signals(work, audio, base)
        test_cli_fake(work, audio)
    finally:
        server.shutdown()
        subprocess.run(["rm", "-rf", work])
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
