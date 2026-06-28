#!/usr/bin/env bash
#
# Compile the LaTeX resume and refresh content/resume.pdf.
#
# The site build (src/main.ts) copies content/resume.pdf to out/res/resume.pdf,
# so updating that file is all that's needed to publish a new resume.
#
# Usage: ./resume/build.sh   (run from anywhere; paths are resolved relative to this script)

set -euo pipefail

# Resolve the directory this script lives in, then the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer pdflatex on PATH; fall back to a known TeX Live location on macOS.
if command -v pdflatex >/dev/null 2>&1; then
  PDFLATEX=pdflatex
elif [ -x /usr/local/texlive/2022/bin/universal-darwin/pdflatex ]; then
  PDFLATEX=/usr/local/texlive/2022/bin/universal-darwin/pdflatex
else
  echo "error: pdflatex not found. Install TeX Live (e.g. 'brew install --cask mactex-no-gui')." >&2
  exit 1
fi

# Build in a temp dir so aux files never clutter the repo.
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Make resume.cls discoverable regardless of the current directory.
export TEXINPUTS="$SCRIPT_DIR:${TEXINPUTS:-}"

# Run twice so hyperref/refs settle.
for _ in 1 2; do
  "$PDFLATEX" -interaction=nonstopmode -halt-on-error \
    -output-directory "$BUILD_DIR" \
    "$SCRIPT_DIR/resume.tex" >/dev/null
done

cp "$BUILD_DIR/resume.pdf" "$REPO_ROOT/content/resume.pdf"
echo "Wrote $REPO_ROOT/content/resume.pdf"
