# stt

One speech-to-text interface for all your transcription tools.

The tools ask stt for a transcript; stt decides which provider does the work.

It transcribes files (`stt transcribe`, `stt.transcribe`) and live audio (`stt stream`, `stt.stream`).

Changing provider is one line in the config.

Adding one is a module in `stt/providers/` (files) or `stt/stream/` (live), plus its entry in that package's `__init__.py`.

## Setup

stt needs Python 3.11 or newer and nothing else for files; live transcription also needs [websockets](https://pypi.org/project/websockets/).

Either install the command with pipx:

```bash
pipx install 'stt[live] @ git+https://github.com/gabrielgydu/stt'
```

Or work from a clone, so that both the command and `import stt` use the working copy:

```bash
./install.sh     # ~/.local/bin/stt, stt.pth (import stt), config skeleton
```

`stt.pth` lives in the user site-packages, which is per Python version.

After a Python minor upgrade (3.14 → 3.15), re-run `./install.sh`; until then `import stt` fails and the tools say so.

Then put your API key in the config (below) and check:

```bash
stt check        # the configured provider, or what is missing
```

## Config

`~/.config/stt/config` (mode 600; `STT_CONFIG` overrides the path), read fresh on every transcription and every live session:

| Key | Default | Meaning |
|-----|---------|---------|
| `provider` | `soniox` | `soniox`, `assemblyai`, or `fake` (for tests: no key, no network) |
| `soniox_api_key` | | Soniox key |
| `assemblyai_api_key` | | AssemblyAI key |
| `languages` | (none: any language) | language hints such as `en,pt`, most likely first (Soniox: hints; AssemblyAI files: expected languages, the first is the fallback; AssemblyAI live: ignored) |
| `keyterms` | | comma-separated names and jargon to get right, most important first (repeats dropped) |
| `soniox_model` | `stt-async-v5` | Soniox async model (files) |
| `assemblyai_model` | `universal-3-5-pro` | AssemblyAI model for files (Universal-2 as fallback) |
| `soniox_stream_model` | `stt-rt-v5` | Soniox realtime model |
| `assemblyai_stream_model` | `universal-3-6-pro` | AssemblyAI realtime model (`universal-streaming*` models get `format_turns`) |
| `fake_fail` | | fake provider only: make it fail (see The fake provider) |
| `fake_finish_delay`, `fake_connect_delay` | | live fake only: seconds to wait (see The fake provider) |

## Checking the setup: `stt check`

```bash
stt check                     # provider and live-transcription state, no network
stt check --json
stt check --provider fake     # as if the config named another provider
```

`stt check` reads the config only; it never touches the network and never prints a key.

Its plain output is `provider: <name> (config: <path>)`, then `stream: ready (<model>)` or `stream: <problem>`.

`stt check --json` prints `{"provider", "config", "ok", "problem", "max_seconds", "stream": {"ok", "problem", "model"}}`.

It exits 0 when file transcription can run, 3 otherwise; the stream line does not change the exit status.

## Transcribing files

### From Python

```python
import stt

problem = stt.config_problem()          # None when transcription can run
transcript = stt.transcribe("rec.mp3", progress=print)
transcript.format()                     # "Speaker A: ..." blocks
transcript.utterances                   # speaker, text, start_ms, end_ms, language

stt.transcribe(audio_bytes)             # bytes work too; nothing is written to disk
stt.transcribe(
    "rec.mp3",
    terms=["Earth, Wind & Fire"],       # replaces the config keyterms ([] = none)
    languages=[],                       # no hints: any language
    spellings=[{"from": ["Sao Paulo", "Saopaulo"], "to": "São Paulo"}],
    deliver=save,                       # gets the transcript before the provider copy is deleted
    timeout=1800,
    keep_on_timeout=False,              # default: a timed-out job is deleted too
)
```

Failures: `stt.ConfigError` (fix the config), `stt.JobFailed` (the provider failed the job; `stt.TooLong` is one, raised before uploading), `stt.rest.ApiError` (`.status`, `.transient`), `OSError` (network; `stt.LeftBehind` is one, see below), `stt.StillRunning` (gave up waiting; the message says whether the job was deleted or may still finish).

A cleanup that Ctrl-C cuts into is finished before the `KeyboardInterrupt` goes on; anything it could not delete is added to the exception as a note.

### From anything else: `stt transcribe`

```bash
stt transcribe rec.mp3                          # "Speaker A: ..." text on stdout
stt transcribe rec.mp3 --json                   # utterances with timestamps and language
stt transcribe rec.mp3 -o rec.txt --no-speakers
stt transcribe - --json < rec.mp3               # audio from stdin, kept in memory
stt transcribe <(ffmpeg ... -f mp3 -)           # a pipe or FIFO works too, read into memory once
stt transcribe rec.mp3 --terms-file terms.txt   # one term per line, commas allowed
stt transcribe rec.mp3 --spellings spellings.json
stt transcribe rec.mp3 --timeout 240 --keep-on-timeout
```

Progress goes to stderr; stdout carries only the transcript.

The transcript is written and flushed before the provider-side copy is deleted, so cleanup never delays it.

Bad arguments are caught before anything is uploaded: a missing input, a directory, or an `-o` that names a directory or a place stt cannot write.

On failure the last stderr line is `stt: <message>`: one line of at most about 300 characters.

Every stderr line, progress included, has every configured API key replaced by `***`.

`STT_DEBUG=1` adds the traceback of an internal error, scrubbed the same way.

| Exit | Meaning | Retry? |
|---|---|---|
| 0 | done; an empty transcript (silence) is a success | |
| 1 | internal error (a bug or an answer stt did not expect), or the transcript could not be written: stdout closed, disk full | no |
| 2 | bad arguments: missing file, a directory, empty stdin, an `-o` that cannot be written, unreadable `--terms-file` or `--spellings` | no |
| 3 | config problem: no key, unknown provider | after fixing the config |
| 4 | the provider refused: bad key, no credit, rejected or too long audio, failed job | no |
| 5 | transient: network, 5xx, 429, after two submit retries | yes |
| 6 | gave up waiting: `--timeout`, or 60 s of failing status checks | yes |
| 130 / 143 | SIGINT / SIGTERM: the job is deleted on the way out; the last line is `stt: cancelled by SIGTERM` (naming anything left behind) | |

A signal that arrives after the transcript was written lets the cleanup finish, then exits 130 / 143 with `stt: SIGTERM arrived after the transcript was written`.

### Spellings

A spellings file is a JSON list like `[{"from": ["Sao Paulo", "Saopaulo"], "to": "São Paulo"}]`, applied after the provider returns, to the text and every utterance.

A form matches regardless of case and only as a whole word: no letter or digit may touch it, accented letters included ("Paraná").

Accents typed as a letter plus a combining mark match precomposed ones and the other way round; the text comes back in Unicode NFC, which looks the same.

The longest form wins, and `to` goes in as written, even several words.

### Waiting and retries

The first status checks come every second for 20 s, then every 3 s (Soniox) or 5 s (AssemblyAI).

A network blip is waited out: stt gives up only after 60 s without a successful status check.

An upload or job creation that fails on the network, with a 5xx or with a 429 is retried twice, after 2 s and 5 s.

Two failures are not retried, because another attempt could leave one more thing at the provider that stt cannot delete (and a second job is billed too): a job creation whose answer was lost (sent, then the connection dropped or timed out), and a Soniox upload that could not be deleted after its job creation failed.

Both end with exit 5 (`stt.LeftBehind` in Python) and say what may be left.

Their message starts with `not retried`, so a caller that retries exit 5 can tell them apart even when a long provider error makes stt clip the line.

With ffprobe installed, a file longer than the provider accepts is refused before uploading (exit 4, naming both lengths).

### Providers for files

| | Soniox (default) | AssemblyAI |
|---|---|---|
| Price per hour, files | $0.10, speakers included | $0.28 with speakers and key terms |
| Longest file | 5 hours | 10 hours |
| Key terms | whole list (~10,000-character context) | first 1,000, none over six words |
| Languages | hints; language identified per token | detection over the hints, first one as fallback; any language without hints |
| Cleanup | file and transcript deleted after each job | transcript and audio deleted after each job |

Soniox keeps uploads until they are deleted, and caps an account at 1,000 files and 2,000 transcriptions.

So every job is deleted when it finishes, fails, times out, or is interrupted: Ctrl-C everywhere, and SIGTERM in the `stt` command.

A Python program that wants the same on SIGTERM installs a handler that raises (as `stt.cli` does); without one, Python's default SIGTERM skips all cleanup.

A cleanup that the signal cuts into is finished first, so the transcript and the file both go.

One narrow window remains: a signal after an upload was sent but before its answer came leaves that upload behind, since stt never learned its id.

Only `--keep-on-timeout` (`keep_on_timeout=True`) leaves a timed-out job in place so it can still be fetched; Soniox purges it after 30 days.

Both providers refuse to delete a job that is still processing; stt then says so in the message ("may still finish") and does not wait for it.

AssemblyAI cannot delete an upload whose transcript was never created; it drops such uploads by itself.

Unverified: with no language hints (the default, or `--languages ''`), AssemblyAI is sent detection without a fallback language, so silence may come back as a failed job (exit 4) instead of an empty success; with hints it completes empty.

## Live transcription (streaming)

stt also transcribes live: raw PCM in, turns of text out, over one provider connection at a time.

stt never reconnects on its own.

A caller that wants to survive a drop opens the next session itself, guided by the error's `fatal`, `routine` and `retry_after`.

Live sessions need websockets (the `[live]` extra, `pip install websockets`, or your distribution's `python-websockets`); `import stt` works without it.

### From Python

```python
from stt import stream

problem = stream.config_problem()       # None when a live session can start (no network)
sess = stream.connect(sample_rate=16000)  # StreamError when it cannot start; one attempt
sess.info                               # provider, model, note, max_burst_seconds, max_session_seconds
sess.send(pcm)                          # raw PCM s16le mono, any length
for event in sess.poll(0.1):            # Partial(turn, text, start_ms)
    ...                                 # Final(turn, text, start_ms, end_ms, speaker, language)
events = sess.finish(timeout=10)        # end of audio: the open turn comes back as a Final
sess.close()                            # cancel: at once, no finals, idempotent
sess.abort()                            # cut the line from a signal handler or another thread
```

`terms` and `languages` default to the config; pass a list (even `[]`) to override.

`speakers=False` is the default for live use, and `endpoint_ms` sets the longest silence before a turn ends.

Key terms are in priority order: when a provider caps them, the first ones are kept and `sess.info.note` says how many of how many were sent.

A Partial carries the open turn's whole text and replaces the previous one.

Exactly one Final closes each turn, turns are numbered 0, 1, 2, …, and finished turns join with one space.

`start_ms` and `end_ms` are milliseconds of audio sent into this session.

When a session ends under the caller, `send`, `poll` and `finish` raise `SessionClosed(error, events)`, where `events` are the ones not returned yet.

If a turn was open when the session ended with an error, `events` closes it with a Final holding its text as last heard (unconfirmed), so every turn that showed a Partial still ends with a Final.

`StreamError.kind` is one of config, auth, credit, rejected, rate_limit, network, server, limit, timeout or internal.

`fatal` means retrying unchanged cannot help, `routine` means the provider's maximum session length was reached (open the next session quietly), and `retry_after` is seconds to wait.

If the provider does not confirm the end of audio within `finish`'s timeout, the open turn as last heard comes back as the Final and `sess.timed_out` says so.

`finish()` on a session that never got audio closes it without asking the provider, since there is no turn to finalize.

A dead network does not hang a session.

websockets' pings (every 20 s, 20 s timeout) catch an idle connection, and audio the network stops acknowledging for 20 s fails a busy one (TCP_USER_TIMEOUT); both come back as kind `network`.

`finish()` returns within about its timeout + 1 s even when the network has stalled.

`abort()` cuts the connection at once without a goodbye: it is the one call that is safe from a signal handler or another thread, a blocked `send` or `finish` then raises `SessionClosed` with no error, and `close()` still follows.

An unreadable config is a `config` problem: `config_problem()` returns `cannot read <path>: …`, and `connect()` raises it as `StreamError(kind="config")`.

### From anything else: `stt stream`

```bash
capture | stt stream [--rate 16000] [--terms T | --terms-file F] [--languages L] [--speakers] \
                     [--endpoint-ms N] [--provider P] [--finish-timeout 5]
```

stdin is raw PCM s16le mono at `--rate`.

It is read from process start, so audio captured while stt connects is not lost.

stdout carries one JSON object per line, flushed at once, and nothing else:

```
{"type": "ready", "provider": "soniox", "model": "stt-rt-v5"}
{"type": "note", "message": "key terms: sent the first 100 of 346 (AssemblyAI streaming takes at most 100)"}
{"type": "partial", "turn": 0, "text": "Bom dia", "start_ms": 120}
{"type": "final", "turn": 0, "text": "Bom dia, Ana.", "start_ms": 120, "end_ms": 2300, "speaker": null, "language": "pt"}
{"type": "error", "kind": "auth", "fatal": true, "routine": false, "retry_after": null, "message": "..."}
{"type": "end"}
```

Close stdin when you are done: the last finals follow, then `end`, and the exit status is 0.

If the provider does not answer within `--finish-timeout`, the last turn as last heard comes as a final, then a `note` saying it was not confirmed, then `end`.

If the session fails mid-way, a turn still open comes first as a final (as last heard), then the error line.

SIGTERM or SIGINT cancels: the provider connection is cut at once, even when a send is stuck on a stalled network, nothing more is printed, and stt exits 143 or 130.

If stdin ends while the network has stalled, stt cannot finish: after 20 s without the network taking audio it prints a `network` error and exits 5.

| Exit | Meaning |
|------|---------|
| 0 | done (`end` printed) |
| 1 | internal error (a bug, audio piling up faster than it can be sent, or stdout closed by its reader) |
| 2 | bad arguments caught by the parser (such as `--rate 0`): usage on stderr, no JSON line |
| 3 | config problem, found before any network: unreadable config, no key, unknown provider, no python-websockets, unreadable `--terms-file` |
| 4 | the provider refused: bad key, no credit, rejected request; retrying will not help |
| 5 | transient: network, provider 5xx, rate limit, session limit; worth another try |
| 130 / 143 | SIGINT / SIGTERM: cancelled, nothing more printed |

On failure the error line goes to stdout, and the last stderr line is `stt: <message>`.

If stdout is already closed, the error line is dropped but the `stt: <message>` line and the exit status stay the same.

No message ever contains an API key.

### Providers for live audio

| | Soniox | AssemblyAI |
|---|---|---|
| Key terms | as many as fit 9,000 characters (the docs put the 8,000-token context limit at about 10,000 characters) | the first 100 |
| Price (2026-10) | $0.12 per hour of audio | $0.45/h for `universal-3-6-pro`, $0.15/h for `universal-streaming-multilingual`, billed while the connection is open, silence included |
| Language hints | sent | ignored |
| Speaker labels | with `--speakers` | not requested |
| Longest session | 300 min | 3 h |
| Faster-than-real-time burst (`max_burst_seconds`) | 10 s (kept low; see below) | 15 s (verified) |

Soniox sends no "session started" frame, so `ready` comes about 1.2 s after connecting, once no error frame has arrived (measured: 0.8 s).

Measured 2026-10-26: Soniox kept a 40 s burst without disconnecting, but worked through it at about real-time speed, so its text came about 35 s later.

A backlog sent at once is therefore transcribed late, not faster, and `finish` after a burst needs a timeout as long as the audio still queued.

Paced live audio is finalized about 0.7 s after the end of audio.

## The fake provider

`provider=fake` (or `--provider fake`) needs no key and never touches the network, so tools that use stt can test against the real `stt` with a throwaway `STT_CONFIG`.

Its output depends only on the audio it receives.

### Files

`stt transcribe` returns `fake transcript of <file name> (<bytes> bytes)` as one utterance: speaker `A` (none with `--no-speakers`), language `en`, start 0; stdin and pipe audio is named `audio`.

| `fake_fail=` | Effect | Exit |
|---|---|---|
| `submit:auth`, `submit:credit`, `submit:rejected`, `submit:limit` | the submit is refused (HTTP 401, 402, 400, or too long) | 4 |
| `submit:server`, `submit:rate_limit`, `submit:network`, `submit:timeout` | transient failure, retried with short pauses, fails every time | 5 |
| `submit:config` | config error | 3 |
| `submit:internal` | unexpected error | 1 |
| `job` | the job fails | 4 |
| `slow:<seconds>` | the job takes that long, to test timeouts | 6 when past `--timeout` |

### Live

Per turn, every 0.5 s of audio gives a partial `fake turn N word`, with one more ` word` each time.

At 2 s of audio the turn ends with the final `fake turn N bytes=<bytes in the turn>`.

At the end of audio, an open turn with audio ends with `fake turn N end total=<bytes received> terms=<count> languages=<a,b>`.

Finals carry speaker `A` with `--speakers`, and the first language hint.

`fake_fail=connect:<kind>` makes connect fail with that kind, and `fake_fail=after:<seconds>:<kind>` ends the session after that much audio (an open turn first comes back as a final, as with the real providers).

`fake_finish_delay=<seconds>` delays the answer to the end of audio, and `fake_connect_delay=<seconds>` slows the connect.

### One `fake_fail` for both

File transcription ignores the live values (`connect:…`, `after:…`), and live transcription ignores the file values (`submit:…`, `job`, `slow:…`).

Any other value makes `stt check` exit 3.

`stt stream` exits 3 too, unless the value at least looks like a file value (`submit:…`, `slow:…`).

## Used by

- [notas](https://github.com/gabrielgydu/notas) dictation, through one `stt stream` per WebSocket session on `/api/dictate`
- [claude-launcher](https://github.com/gabrielgydu/claude-launcher) voice clips, through `stt transcribe --json --no-speakers -` with the clip on stdin
- [tubereader](https://github.com/gabrielgydu/tubereader), through `stt transcribe --json --languages ""`, with `--terms ""` or a vocabulary's terms and spellings files
- [nvim-dictation](https://github.com/gabrielgydu/nvim-dictation), through `stt stream` spawned by its Deno sidecar when `provider = 'stt'`

## Tests

```bash
python3 tests/test_stt.py      # file providers, fake, spellings and `stt transcribe` / `stt check`, against a local mock API
python3 tests/test_stream.py   # both realtime providers against local websocket mocks, a stalled network, the fake, and `stt stream`
```

No network, no credit, never the real config.

The duration-guard checks need ffmpeg, and `tests/test_stream.py` needs websockets.

## License

MIT
