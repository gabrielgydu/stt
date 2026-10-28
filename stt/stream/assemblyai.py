"""
AssemblyAI real-time transcription (v3, wss://streaming.assemblyai.com/v3/ws),
ported from audiorecorder's --stream.

The session is configured in the URL: the speech model, the sample rate, the
key terms as a JSON list, and format_turns for the older Universal-Streaming
models (the Pro ones always punctuate; the older ones only when asked, and then
send each finished turn twice, formatted second). The raw key is the
Authorization header. Begin opens the session, Turn frames carry the open
turn's whole text, Terminate asks for the end and Termination confirms it.

Framing is stt's job here: the API rejects chunks outside 50-1000 ms, so small
sends are buffered until 50 ms, big ones are split, and only the very last
chunk is padded with silence at finish (padding mid-session would shift the
clock). `languages` are ignored, as in batch; speaker labels are not asked for.
"""

import json
import time
import urllib.parse

from . import WsSession

DEFAULT_MODEL = "universal-3-6-pro"
KEYTERMS_MAX = 100        # a longer list fails every session (3006, fatal)
# "Too many concurrent sessions" (1008) is the per-minute limit on new sessions.
RATE_LIMIT_RETRY = 60


class AssemblyAI(WsSession):
    name = "assemblyai"
    title = "AssemblyAI"
    url = "wss://streaming.assemblyai.com/v3/ws"
    model_key = "assemblyai_stream_model"
    default_model = DEFAULT_MODEL
    keepalive_frame = None    # v3 has no keepalive message
    rate_limit_retry = RATE_LIMIT_RETRY
    # The API closes a session fed faster than real time (3007) without saying
    # by how much; a 15 s burst was verified to pass (audiorecorder).
    max_burst_seconds = 15
    max_session_seconds = 3 * 3600   # then 3008, routine

    def __init__(self, *args):
        super().__init__(*args)
        self.min_chunk = -(-self.sample_rate // 20) * 2   # 50 ms in whole samples
        self.max_chunk = self.sample_rate * 2             # 1000 ms
        self._buf = b""
        self._error_frame = None
        self._format_turns = self.model.startswith("universal-streaming")
        if self.speakers:
            self.notes.append("speaker labels are not asked for in AssemblyAI streaming")

    def _cap_terms(self, terms):
        if len(terms) <= KEYTERMS_MAX:
            return terms
        self.notes.append(f"key terms: sent the first {KEYTERMS_MAX} of {len(terms)} "
                          f"(AssemblyAI streaming takes at most {KEYTERMS_MAX})")
        return terms[:KEYTERMS_MAX]

    def _connect_url(self):
        params = {"speech_model": self.model, "sample_rate": str(self.sample_rate)}
        if self._format_turns:
            params["format_turns"] = "true"
        if self.endpoint_ms is not None:
            params["max_turn_silence"] = str(int(self.endpoint_ms))
        if self.terms:
            params["keyterms_prompt"] = json.dumps(self.terms)
        return f"{self.url}?{urllib.parse.urlencode(params)}"

    def _headers(self):
        return {"Authorization": self.config.get("assemblyai_api_key")}

    def _accept(self, deadline):
        """Wait for Begin. A bad key does not fail the handshake: the socket
        opens and the first frame is an error."""
        try:
            msg = json.loads(self._ws.recv(timeout=max(0, deadline - time.monotonic())))
        except TimeoutError:
            raise self._fail("AssemblyAI did not start the session in time", "timeout") from None
        if msg.get("type") == "Error":
            raise self._code_error(msg.get("error_code"), msg.get("error") or "")
        if msg.get("type") != "Begin":
            raise self._fail(f"AssemblyAI answered {msg.get('type')!r} instead of Begin", "internal")

    def _send_audio(self, pcm):
        self._buf += pcm
        while len(self._buf) >= self.min_chunk:
            n = min(len(self._buf), self.max_chunk)
            n -= n % 2
            self._ws.send(self._buf[:n])
            self._buf = self._buf[n:]

    def _send_end(self):
        if self._buf:
            last = self._buf + b"\0" * (len(self._buf) % 2)
            last += b"\0" * max(0, self.min_chunk - len(last))
            self._buf = b""
            self._ws.send(last)
        self._ws.send(json.dumps({"type": "Terminate"}))

    # -- frames ----------------------------------------------------------------

    def _handle(self, msg):
        kind = msg.get("type")
        if kind == "Error":
            # The close that follows only says "See Error message for details".
            self._error_frame = msg
        elif kind == "Turn":
            words = msg.get("words") or []
            start = words[0].get("start") if words else None
            if msg.get("end_of_turn") and (msg.get("turn_is_formatted") or not self._format_turns):
                self._final(msg.get("transcript"), start, words[-1].get("end") if words else None,
                            None, msg.get("language_code"))
            else:
                self._partial(msg.get("transcript"), start)
        elif kind == "Termination":
            if not self._ending:
                # Not asked for: the close that follows says why (an Error
                # frame, or 3008 at the 3 h limit), as audiorecorder read it.
                return False
            self._flush_open_turn()
            return True
        return False

    def _closed_error(self, exc):
        """The error frame sent before the close is preferred; a dropped
        network leaves neither."""
        rcvd = exc.rcvd
        code = rcvd.code if rcvd else None
        if self._error_frame:
            return self._code_error(self._error_frame.get("error_code", code), self._error_frame.get("error") or "")
        return self._code_error(code, (rcvd.reason if rcvd else "") or "connection to AssemblyAI lost")

    def _code_error(self, code, text):
        text = text or f"AssemblyAI error {code}"
        retry = None
        if code == 1008:
            if "Too many concurrent sessions" in text:
                kind, retry = "rate_limit", RATE_LIMIT_RETRY
            elif any(word in text.lower() for word in ("insufficient funds", "balance", "credit")):
                kind = "credit"
            else:
                kind = "auth"
        elif code == 3006:
            kind = "rejected"
        elif code == 3008:
            kind = "limit"
        elif code in (None, 1000, 1001, 1006):
            kind = "network"
        else:
            kind = "server"   # 1011 and anything else: retrying may work
        return self._fail(text, kind, code=code, retry_after=retry)
