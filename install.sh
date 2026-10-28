#!/usr/bin/env bash
# Make stt importable and runnable for the user:
#   - ~/.local/bin/stt -> bin/stt
#   - stt.pth in the user site-packages, so `import stt` works from any Python
#     tool (re-run after a Python minor upgrade: the site dir is per version)
#   - ~/.config/stt/config skeleton (mode 600) when there is none yet
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

mkdir -p "$HOME/.local/bin"
ln -sfn "$ROOT/bin/stt" "$HOME/.local/bin/stt"
echo "→ $HOME/.local/bin/stt -> $ROOT/bin/stt"

SITE="$(python3 -c 'import site; print(site.getusersitepackages())')"
mkdir -p "$SITE"
echo "$ROOT" > "$SITE/stt.pth"
echo "→ $SITE/stt.pth"

CONFIG="${STT_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/stt/config}"
if [[ ! -e "$CONFIG" ]]; then
  mkdir -p "$(dirname "$CONFIG")"
  (umask 077; cat > "$CONFIG" <<'EOF'
# Speech-to-text config for every transcription tool (see the stt README).
provider=soniox
soniox_api_key=
assemblyai_api_key=
# Language hints, most likely first (e.g. en,pt); empty = any language.
languages=
keyterms=
EOF
  )
  echo "→ created $CONFIG; put the API key in it"
fi

python3 -c 'import stt' && echo "✓ import stt works"
"$HOME/.local/bin/stt" check || true
