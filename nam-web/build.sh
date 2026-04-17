#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building nam-wasm..."
wasm-pack build "$ROOT_DIR/nam-wasm" --target web --release

echo "Copying package to nam-web..."
rm -rf "$SCRIPT_DIR/pkg"
cp -r "$ROOT_DIR/nam-wasm/pkg" "$SCRIPT_DIR/pkg"

echo ""
echo "Done. To run the demo:"
echo "  cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "  Open http://localhost:8080"
