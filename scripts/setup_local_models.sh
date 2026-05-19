#!/usr/bin/env bash
# Pull the local LLM models LaunchLens uses on the 8 GB VRAM dev hardware.
# Q4 quantized GGUF, all comfortably fit alongside a 4K context window.
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found. Install from https://ollama.com/download then re-run."
    exit 1
fi

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama server not running. Start it with: ollama serve &"
    echo "Then re-run this script."
    exit 1
fi

DEFAULT_MODEL="${OLLAMA_DEFAULT_MODEL:-qwen2.5:3b-instruct-q4_K_M}"
FAST_MODEL="${OLLAMA_FAST_MODEL:-gemma2:2b-instruct-q4_K_M}"

echo "Pulling default multilingual model: $DEFAULT_MODEL"
ollama pull "$DEFAULT_MODEL"

echo "Pulling fast/light model: $FAST_MODEL"
ollama pull "$FAST_MODEL"

# Sarvam-1 2B is not always available as an upstream Ollama tag.
# Try to pull; if it fails, document the manual GGUF import path.
SARVAM_TAG="${OLLAMA_INDIC_MODEL:-sarvam-1:2b}"
if ollama pull "$SARVAM_TAG" 2>/dev/null; then
    echo "Pulled Indic model: $SARVAM_TAG"
else
    cat <<EOF

NOTE: '$SARVAM_TAG' is not available on the upstream Ollama registry.
You can either:
  1. Use $DEFAULT_MODEL for Indic routes (already capable in Hindi/Tamil/Telugu/Bengali).
  2. Manually import a Sarvam GGUF:
       ollama create sarvam-1 -f Modelfile     # with a Modelfile pointing at the GGUF
LaunchLens config 'ollama_indic_model' falls back to '$DEFAULT_MODEL' until then.
EOF
fi

echo ""
echo "Installed models:"
ollama list
