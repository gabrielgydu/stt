"""Exit statuses of the `stt` command, shared by every subcommand.

Callers that are not Python tell failures apart by these: a config problem
needs a person, a fatal provider error will fail the same way again, and a
transient one is worth one more try.
"""

OK = 0
INTERNAL = 1    # a bug or an answer stt did not expect; the message says what
USAGE = 2       # bad arguments (argparse's own status)
CONFIG = 3      # no key, unknown provider, missing library: fix the config
FATAL = 4       # the provider refused: bad key, no credit, rejected request or audio, job failed
TRANSIENT = 5   # network, provider 5xx, rate limit, session limit: retrying may work
TIMEOUT = 6     # gave up waiting for a batch job
SIGINT = 130
SIGTERM = 143
