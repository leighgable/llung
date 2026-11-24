#!/usr/bin/env bash
# Serve SenseNova U1.5 8B (sensenova_u1 arch) with the pinned
# sensenova-loop llama.cpp fork (llama-cpp-gpu in the devShell) on the
# llung local backend port 9931 — in ROUTER MODE, next to SmolVLM, the
# history-compressor model for llamacpp agents. Requests pick a model
# by the `model` field: preset `sensenova-9b` (the agent) and preset
# `smolvlm` (the compressor). SmolVLM is fetched from Hugging Face on
# first start (cached under LLAMA_CACHE / ~/.cache/llama.cpp).
#
# Weights are NOT committed. Get them either way:
#
#   A. Convert from HF (the fork converts `sensenova/SenseNova-U1.5-8B-MoT`
#      — language + vision tower), from the fork source:
#        python conversion/sensenova_u1.py <hf-checkout-or-repo> \
#            --outfile sensenova-u1.5.gguf [--mmproj mmproj.gguf]
#      (the fork source lives in the nix store; find it with
#       `nix-store -qR $(readlink -f $(which llama-server)) | grep source`)
#
#   B. Point this script at GGUFs you already have:
#        LLUNG_SENSENOVA_GGUF=/path/to/model.gguf \
#        LLUNG_SENSENOVA_MMPROJ=/path/to/mmproj.gguf \
#        scripts/serve-sensenova.sh
#
# Env:
#   LLUNG_SENSENOVA_GGUF   model GGUF (default: ./models then ~/models
#                          scan for *ensenova*.gguf)
#   LLUNG_SENSENOVA_MMPROJ mmproj GGUF; omit to serve TEXT-ONLY
#   LLAMA_PORT             (default 9931)
#   NGL                    GPU layers (default 99 = all)
#   LLUNG_SMOLVLM_HF       SmolVLM HF repo[:quant]
#                          (default ggml-org/SmolVLM2-2.2B-Instruct-GGUF)
#   LLUNG_SMOLVLM_GGUF     local SmolVLM GGUF instead of the HF repo
#   LLUNG_SMOLVLM_CTX      SmolVLM context size (default 8192)
#
# Vision notes: the mmproj needs the mtmd build — llama-cpp-gpu has it.
# An 8B Q4 (~5.5 GB) + vision tower fits in ~7-8 GB vRAM; on the 890M
# iGPU (512 GB/s unified memory) it runs, slowly — expect a minute for
# the first token. Text-only serving halves that.
set -euo pipefail

PORT="${LLAMA_PORT:-9931}"
NGL="${NGL:-99}"

MODEL="${LLUNG_SENSENOVA_GGUF:-}"
if [[ -z "$MODEL" ]]; then
    for dir in ./models "$HOME/models"; do
        found=$(ls "$dir"/*[sS]ensenova*.gguf 2>/dev/null | grep -v -i mmproj | head -1 || true)
        if [[ -n "$found" ]]; then MODEL="$found"; break; fi
    done
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
    echo "error: no SenseNova GGUF found. Set LLUNG_SENSENOVA_GGUF." >&2
    echo "       (see the header of this script for conversion notes)" >&2
    exit 1
fi
MODEL="$(readlink -f "$MODEL")"

MMPROJ="${LLUNG_SENSENOVA_MMPROJ:-}"
# Preset names are the model ids llung requests: they must match
# LLUNG_LLAMA_MODEL / LLUNG_SMOLVLM_MODEL (defaults below).
AGENT_ID="${LLUNG_LLAMA_MODEL:-sensenova-9b}"
SMOL_ID="${LLUNG_SMOLVLM_MODEL:-smolvlm}"
PRESET="$(mktemp --suffix=.ini "${TMPDIR:-/tmp}/llung-models.XXXXXX")"
trap 'rm -f "$PRESET"' EXIT

{
    echo "version = 1"
    echo
    echo "[*]"
    echo "n-gpu-layers = $NGL"
    echo "jinja = true"
    echo
    echo "[$AGENT_ID]"
    echo "model = $MODEL"
    if [[ -n "$MMPROJ" && -f "$MMPROJ" ]]; then
        echo "mmproj = $(readlink -f "$MMPROJ")"
    fi
    echo "load-on-startup = true"
    echo
    # History compression is text-only: skip SmolVLM's vision tower.
    echo "[$SMOL_ID]"
    if [[ -n "${LLUNG_SMOLVLM_GGUF:-}" ]]; then
        echo "model = $(readlink -f "$LLUNG_SMOLVLM_GGUF")"
    else
        echo "hf-repo = ${LLUNG_SMOLVLM_HF:-ggml-org/SmolVLM2-2.2B-Instruct-GGUF}"
    fi
    echo "c = ${LLUNG_SMOLVLM_CTX:-8192}"
    echo "no-mmproj = true"
    echo "load-on-startup = true"
} > "$PRESET"

if [[ -n "$MMPROJ" && -f "$MMPROJ" ]]; then
    echo "serving WITH vision: $(basename "$MODEL") + $(basename "$MMPROJ") as '$AGENT_ID'"
else
    echo "serving text-only: $(basename "$MODEL") as '$AGENT_ID' (set LLUNG_SENSENOVA_MMPROJ for vision)"
fi
echo "history compressor: SmolVLM as '$SMOL_ID'"

# gfx1150 (Radeon 890M iGPU) needs the override unless ROCm knows it.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.0}"

# No exec: the EXIT trap removes the preset file afterwards.
llama-server --port "$PORT" --models-preset "$PRESET"
