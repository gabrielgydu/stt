"""Minimal JSON-over-HTTP on the standard library, shared by the providers.

Error messages from here can end up on a terminal, in a service log or on a
phone screen, so they are kept to one line of about 300 characters and never
carry the API key, whatever the provider echoes back.
"""

import http.client
import json
import urllib.error
import urllib.request
import uuid

# Per-request socket timeout. Without one, urlopen blocks forever and a single
# stalled poll hangs the whole run even when the job finished server-side.
API_TIMEOUT = 30
UPLOAD_TIMEOUT = 300
# Cleanup runs after the transcript is out (or while the process is being
# stopped), so a stalled DELETE must not hold anything up for long.
DISCARD_TIMEOUT = 10

MESSAGE_LIMIT = 300


class ApiError(RuntimeError):
    """A request the provider answered with an HTTP error."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

    @property
    def transient(self):
        return self.status >= 500 or self.status in (408, 429)


class AnswerLost(ConnectionError):
    """The request went out but its answer never came: the connection dropped,
    timed out or broke (TLS) while waiting for it or reading it. A network
    failure like any other, except that the provider may already have acted on
    the request, so repeating one that creates a job may create a second."""


def is_transient(error):
    """Whether retrying the same request later may work: a 5xx, 408 or 429, or
    a network failure. A local file error (missing audio) is not one."""
    if isinstance(error, ApiError):
        return error.transient
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))


def clip(text, limit=MESSAGE_LIMIT):
    """`text` on one line, cut to about `limit` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def scrub(text, secrets):
    """`text` with every secret replaced by ***."""
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _detail(raw):
    """The provider's human message from an error body, else the body itself."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(data, dict):
        for key in ("message", "error_message", "error"):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key]
    return raw


def request(method, url, headers=None, body=None, timeout=API_TIMEOUT, secrets=()):
    """Send a request and return the parsed JSON body (None when empty).

    `body` is bytes, or anything else is sent as JSON. Raises ApiError with
    the provider's error message, clipped and with `secrets` scrubbed, on an
    HTTP error status. Network failures are OSError: URLError when the request
    could not be sent, AnswerLost when it was sent and the answer never came.
    """
    headers = dict(headers or {})
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except (OSError, http.client.HTTPException):
            raw = b""
        # Scrubbed before it is cut, so a key on the cut cannot leave a prefix.
        detail = _detail(raw.decode("utf-8", errors="replace"))
        message = clip(scrub(f"HTTP {e.code} {e.reason}: {detail}", secrets))
        raise ApiError(e.code, message) from None
    except urllib.error.URLError:
        raise  # urllib wraps only what fails while connecting or sending
    except (OSError, http.client.HTTPException) as e:
        # Anything else failed after the request went out: a dropped or timed
        # out answer, a TLS error, IncompleteRead. One class for all of them,
        # so callers need not know http.client or ssl.
        detail = scrub(f"{type(e).__name__}: {e}", secrets)
        raise AnswerLost(f"{method} {url}: sent, but no answer came ({clip(detail, 120)})") from None
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError:
        start = scrub(raw.decode("utf-8", errors="replace"), secrets)
        raise ValueError(f"{method} {url}: the answer was not JSON: {clip(start, 100)}") from None


def multipart(field, filename, data):
    """A multipart/form-data body holding one file, and its Content-Type."""
    boundary = uuid.uuid4().hex
    name = filename.replace('"', "").replace("\r", "").replace("\n", "")
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + data + tail, f"multipart/form-data; boundary={boundary}"
