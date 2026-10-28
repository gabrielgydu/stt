"""
Soniox real-time transcription
(https://soniox.com/docs/api-reference/stt/websocket-api).

The key goes in the Authorization header of the upgrade, never in the start
message (deprecated there, refused from 2027-01-15). The start message asks for
raw 16-bit PCM, endpoint detection (turns come from its <end> tokens),
language identification, and the key terms as context.terms.

Soniox streams tokens, not turns. Final tokens arrive once; the non-final ones
are re-sent whole in every response. A turn is the final tokens up to an <end>
(endpoint) or <fin> (finalize) marker or, with speakers on, up to a change of
speaker; its Partial is those final tokens plus the current non-final tail.
Token times are ms of the stream's audio, which is the session clock.
"""

import json
import time
from collections import Counter

from . import WsSession

DEFAULT_MODEL = "stt-rt-v5"
# Soniox sends nothing when it accepts a stream; a bad key or a refused start
# message comes back as an error frame within about a second.
ACCEPT_GRACE_SECONDS = 1.2
# A context over 8,000 tokens fails the session (400, fatal). The docs put
# that at about 10,000 characters; key terms are names and jargon, which split
# into more tokens than prose, so stt sends at most 9,000 characters of them
# (each term counted with one separator).
CONTEXT_CHARS = 9000
# Slots held by dropped streams free within 20 s (the keepalive timeout).
RATE_LIMIT_RETRY = 20
ENDPOINT_MIN_MS, ENDPOINT_MAX_MS = 500, 3000
MARKERS = ("<end>", "<fin>")


class Soniox(WsSession):
    name = "soniox"
    title = "Soniox"
    url = "wss://stt-rt.soniox.com/transcribe-websocket"
    model_key = "soniox_stream_model"
    default_model = DEFAULT_MODEL
    keepalive_frame = json.dumps({"type": "keepalive"})
    rate_limit_retry = RATE_LIMIT_RETRY
    # The docs want "real-time or near real-time speed" and warn that
    # "prolonged bursts may disconnect". Measured 2026-10-26: a 40 s burst
    # stayed connected but was worked through at about real time (final
    # ~35 s after ready), so a backlog is transcribed late, not faster.
    # Kept below AssemblyAI's verified 15 s.
    max_burst_seconds = 10
    max_session_seconds = 300 * 60   # fixed; 413 max_duration_reached after

    def __init__(self, *args):
        super().__init__(*args)
        self._finals = []   # final tokens of the open turn
        self._tail = []     # the latest non-final tokens
        self._endpoint_delay = None
        if self.endpoint_ms is not None:
            wanted = int(self.endpoint_ms)
            self._endpoint_delay = min(max(wanted, ENDPOINT_MIN_MS), ENDPOINT_MAX_MS)
            if self._endpoint_delay != wanted:
                self.notes.append(f"endpoint delay {wanted} ms sent as {self._endpoint_delay} ms "
                                  f"(Soniox takes {ENDPOINT_MIN_MS}-{ENDPOINT_MAX_MS})")

    def _cap_terms(self, terms):
        kept, used = [], 0
        for term in terms:
            used += len(term) + 1
            if used > CONTEXT_CHARS:
                break
            kept.append(term)
        if len(kept) < len(terms):
            self.notes.append(f"key terms: sent the first {len(kept)} of {len(terms)} "
                              f"(Soniox's context limit)")
        return kept

    def _headers(self):
        return {"Authorization": f"Bearer {self.config.get('soniox_api_key')}"}

    def _start_message(self):
        msg = {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            "enable_language_identification": True,
            "enable_speaker_diarization": self.speakers,
            "enable_endpoint_detection": True,
        }
        if self.languages:
            msg["language_hints"] = self.languages
        if self._endpoint_delay is not None:
            msg["max_endpoint_delay_ms"] = self._endpoint_delay
        if self.terms:
            msg["context"] = {"terms": self.terms}
        return msg

    def _accept(self, deadline):
        """No Begin frame here: accepted when no error frame came within the
        grace, or when the first response did."""
        self._ws.send(json.dumps(self._start_message()))
        grace = max(0, min(ACCEPT_GRACE_SECONDS, deadline - time.monotonic()))
        try:
            message = self._ws.recv(timeout=grace)
        except TimeoutError:
            return
        self._handle(json.loads(message))

    def _send_audio(self, pcm):
        self._ws.send(pcm)

    def _send_end(self):
        self._ws.send("")   # an empty text frame; an empty binary one is just no audio

    # -- frames ----------------------------------------------------------------

    def _handle(self, msg):
        if msg.get("error_code") is not None or msg.get("error_type"):
            raise self._frame_error(msg)
        tail = []
        for token in msg.get("tokens") or []:
            if not token.get("is_final"):
                tail.append(token)
            elif token.get("text") in MARKERS:
                self._end_turn()
            else:
                if self.speakers and self._speaker_changed(token):
                    self._end_turn()
                self._finals.append(token)
        self._tail = tail
        if msg.get("finished"):
            self._end_turn(with_tail=True)
            return True
        tokens = self._finals + self._tail
        self._partial("".join(t.get("text") or "" for t in tokens),
                      tokens[0].get("start_ms") if tokens else None)
        return False

    def _speaker_changed(self, token):
        if not self._finals:
            return False
        before, now = self._finals[-1].get("speaker"), token.get("speaker")
        return before is not None and now is not None and before != now

    def _end_turn(self, with_tail=False):
        tokens = self._finals + (self._tail if with_tail else [])
        self._finals = []
        if with_tail:
            self._tail = []
        speakers = [t["speaker"] for t in tokens if t.get("speaker") is not None]
        languages = [t["language"] for t in tokens if t.get("language")]
        self._final(
            "".join(t.get("text") or "" for t in tokens),
            tokens[0].get("start_ms") if tokens else None,
            tokens[-1].get("end_ms") if tokens else None,
            self._letter(speakers[0]) if speakers else None,
            Counter(languages).most_common(1)[0][0] if languages else None,
        )

    def _flush_open_turn(self):
        if self._turn_open or self._finals or self._tail:
            self._end_turn(with_tail=True)

    def _frame_error(self, msg):
        """Soniox's error frame (sent right before it closes with 1000) as a
        StreamError. error_type is the stable field; the code covers the rest."""
        code = msg.get("error_code")
        kind_name = msg.get("error_type") or ""
        text = msg.get("error_message") or kind_name or f"Soniox error {code}"
        retry = None
        if kind_name in ("temp_api_key_session_expired", "max_duration_reached") or code == 413:
            kind = "limit"
        elif code == 400:
            kind = "rejected"
        elif code in (401, 403):
            kind = "auth"
        elif code == 402:
            kind = "credit"
        elif code == 408:
            kind = "network"   # too slow on audio / timed out: a new stream may work
        elif code == 429:
            kind, retry = "rate_limit", RATE_LIMIT_RETRY
        elif isinstance(code, int) and 400 <= code < 500:
            kind = "rejected"
        else:
            kind = "server"
        return self._fail(text, kind, code=code, retry_after=retry)
