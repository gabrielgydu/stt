"""
The one config every transcription tool reads: which provider, its API key,
the language hints and the key terms.

    ~/.config/stt/config          (STT_CONFIG overrides the path)

    provider=soniox               # soniox (default) | assemblyai | fake (tests)
    soniox_api_key=...
    assemblyai_api_key=...
    languages=en,pt               # hints, first = most likely (AssemblyAI: its fallback); none by default
    keyterms=Kubernetes, ...     # names and jargon to get right

Plain key=value lines, '#' starts a comment. It is read fresh on every
call, so editing it takes effect without restarting anything.
"""

import os

DEFAULT_PROVIDER = "soniox"
DEFAULT_LANGUAGES = ""  # no hints: any language


def path():
    """Where the config lives: $STT_CONFIG, else $XDG_CONFIG_HOME/stt/config."""
    override = os.environ.get("STT_CONFIG")
    if override:
        return os.path.expanduser(override)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "stt", "config")


def load(config_path=None):
    """The config as a dict of strings; empty when the file does not exist."""
    config_path = config_path or path()
    config = {}
    try:
        with open(config_path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return config
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def split_list(raw):
    """A comma-separated setting as a list, without empty entries."""
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def dedupe(items):
    """The list without repeats, first occurrence kept (exact match)."""
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def read_terms_file(file_path):
    """Key terms from a file, one per line, so a term may contain commas.
    Blank lines and lines starting with '#' are skipped."""
    with open(os.path.expanduser(file_path)) as f:
        lines = [line.strip() for line in f.read().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def api_keys(config):
    """Every configured *_api_key value, for scrubbing from messages."""
    return [value for key, value in config.items() if key.endswith("_api_key") and value]


def provider_name(config):
    return (config.get("provider") or DEFAULT_PROVIDER).strip().lower()


def keyterms(config):
    return split_list(config.get("keyterms"))


def languages(config):
    return split_list(config.get("languages", DEFAULT_LANGUAGES))
