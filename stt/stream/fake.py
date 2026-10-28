"""
A provider for consumer tests: no network, no key, and events that depend only
on the audio received, so a test can drive the real `stt stream` (or
stt.stream.connect) and know exactly what comes back.

    provider=fake                 # in the STT_CONFIG file a test writes
    fake_fail=connect:auth        # connect() raises StreamError(kind="auth")
    fake_fail=after:3:network     # the session ends with that error after 3 s of audio
    fake_finish_delay=2           # finish() answers after 2 s (to test your timeouts)
    fake_connect_delay=0.5        # connect() takes this long

Per turn, every 0.5 s of audio: Partial "fake turn N word" with one " word"
per 0.5 s so far. At 2 s: Final "fake turn N bytes=<bytes in the turn>", and
turn N+1 starts. finish(): Final "fake turn N end total=<bytes received>
terms=<count> languages=<a,b>" for an open turn that has audio. Finals carry
speaker "A" with speakers on, and the first language hint. When fake_fail ends
the session, an open turn first comes back as a Final as last heard, as with
the real providers.

The error kinds behave as for the real providers (fatal for config, auth,
credit, rejected; routine for limit); rate_limit asks for a 1 s wait.

The batch fake (stt.providers.fake) reads the same fake_fail key; its values
(submit:..., job, slow:...) are ignored here, as ours are there.
"""

import threading
import time

from . import FATAL_KINDS, Session

KINDS = FATAL_KINDS + ("rate_limit", "network", "server", "limit", "timeout", "internal")


def _settings(config):
    """(fail_at_connect, fail_after_seconds, fail_kind, finish_delay, connect_delay);
    ValueError naming the bad setting."""
    fail = (config.get("fake_fail") or "").strip()
    at_connect, after, kind = False, None, None
    # File transcription's fake reads the same key; its values are not ours.
    if fail == "job" or fail.partition(":")[0] in ("submit", "slow"):
        fail = ""
    if fail:
        parts = fail.split(":")
        try:
            if parts[0] == "connect" and len(parts) == 2:
                at_connect, kind = True, parts[1]
            elif parts[0] == "after" and len(parts) == 3:
                after, kind = float(parts[1]), parts[2]
            else:
                raise ValueError
        except ValueError:
            raise ValueError(f"fake_fail must be connect:<kind> or after:<seconds>:<kind>, not {fail!r}") from None
        if kind not in KINDS:
            raise ValueError(f"fake_fail: unknown kind {kind!r}; known: {', '.join(KINDS)}")
    delays = []
    for key in ("fake_finish_delay", "fake_connect_delay"):
        try:
            delays.append(float(config.get(key) or 0))
        except ValueError:
            raise ValueError(f"{key} must be a number of seconds, not {config.get(key)!r}") from None
    return at_connect, after, kind, delays[0], delays[1]


class Fake(Session):
    name = "fake"
    title = "fake"
    default_model = "fake"
    needs_key = False
    needs_websockets = False
    max_burst_seconds = 15
    max_session_seconds = 3600

    @classmethod
    def model_for(cls, config):
        return "fake"

    @classmethod
    def problem(cls, config):
        try:
            _settings(config)
        except ValueError as e:
            return str(e)
        return None

    def __init__(self, *args):
        super().__init__(*args)
        (self._fail_at_connect, self._fail_after, self._fail_kind,
         self._finish_delay, self._connect_delay) = _settings(self.config)
        self._step = self.sample_rate            # bytes in 0.5 s
        self._received = 0
        self._turn_start = 0                     # where the open turn began, in bytes
        self._turn_bytes = 0
        self._timer = None

    def _failure(self, what):
        kind = self._fail_kind
        return self._fail(f"fake {kind} failure {what}", kind,
                          retry_after=1.0 if kind == "rate_limit" else None)

    def _start(self, timeout):
        if self._connect_delay:
            time.sleep(min(self._connect_delay, timeout))
            if self._connect_delay > timeout:
                raise self._fail(f"fake: no answer within {timeout:g} s", "timeout")
        if self._fail_at_connect:
            raise self._failure("at connect")

    def _send_audio(self, pcm):
        left = len(pcm)
        limit = None if self._fail_after is None else int(self._fail_after * 2 * self.sample_rate)
        while left:
            if limit is not None and self._received >= limit:
                self._break_off()
                return
            take = min(left, self._step - self._turn_bytes % self._step)
            if limit is not None:
                take = min(take, limit - self._received)
            left -= take
            self._received += take
            self._turn_bytes += take
            if self._turn_bytes % self._step:
                continue
            start, end = self._ms(self._turn_start), self._ms(self._turn_start + self._turn_bytes)
            words = self._turn_bytes // self._step
            if words == 4:
                self._final(f"fake turn {self._turn} bytes={self._turn_bytes}", start, end,
                            *self._labels())
                self._turn_start += self._turn_bytes
                self._turn_bytes = 0
            else:
                self._partial(f"fake turn {self._turn}" + " word" * words, start)
        if limit is not None and self._received >= limit:
            self._break_off()

    def _break_off(self):
        """fake_fail=after:...: the session ends under the caller, the open
        turn coming back as last heard, as with the real providers."""
        self._flush_open_turn()
        self._end(self._failure(f"after {self._fail_after:g} s of audio"))

    def _flush_open_turn(self):
        if self._turn_open:
            text, start = self._latest
            self._final(text, start, self._ms(self._turn_start + self._turn_bytes), *self._labels())

    def _labels(self):
        return ("A" if self.speakers else None), (self.languages[0] if self.languages else None)

    def _send_end(self):
        if self._finish_delay > 0:
            self._timer = threading.Timer(self._finish_delay, self._answer_end)
            self._timer.daemon = True
            self._thread = self._timer
            self._timer.start()
        else:
            self._answer_end()

    def _answer_end(self):
        if self._stopping:
            return
        if self._turn_bytes:
            self._final(
                f"fake turn {self._turn} end total={self._received} terms={len(self.terms)} "
                f"languages={','.join(self.languages)}",
                self._ms(self._turn_start), self._ms(self._turn_start + self._turn_bytes), *self._labels(),
            )
        self._finished = True
        self._end()

    def _stop_transport(self):
        if self._timer is not None:
            self._timer.cancel()
