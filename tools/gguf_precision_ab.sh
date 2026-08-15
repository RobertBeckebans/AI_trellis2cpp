#!/usr/bin/env bash
# A/B the mesh topology of one generation across weight precisions.
#
# The question: f16 weights leave ~1e-3 of noise on the subdivision logits the
# shape decoder thresholds at zero, f32 leaves ~1e-6. Does that show up as
# ragged geometry — more non-manifold and boundary edges — or not?
#
# Three variants, same image, same seed, same tier:
#   f16       everything as shipped
#   dec32     only the shape decoder in f32 — it owns the to_subdiv head, so if
#             the threshold noise is what matters this alone should move it
#   all32     the whole geometry path in f32, as the upper bound
#
# Reads the "[mesh] ... boundary edges ... non-manifold edges" line the C-ABI
# prints under TRELLIS2_TIMING. Comparing runs is only meaningful because the
# sampler is reproducible again (exact attention + no CUDA graphs); before that
# two runs of the same seed differed for unrelated reasons.
#
# Usage: tools/gguf_precision_ab.sh [quality] [seed]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUALITY="${1:-1024}"
SEED="${2:-1234}"
PORT=8743
IMAGE="$ROOT/dumps/fixture_rgba.png"
OUT="$ROOT/build/precision-ab"

export TRELLIS2_TIMING=1
# Both, exactly as start_server.bat does: trellis2.dll pulls in the ggml DLLs
# beside it and the ROCm runtime, and the loader searches PATH for neither
# unless it is told to.
export PATH="/c/Program Files/AMD/ROCm/6.4/bin:$ROOT/build/bin/Release:$PATH"

[ -f "$IMAGE" ] || { echo "missing $IMAGE (run scripts/dump_dino_reference.py)"; exit 1; }
mkdir -p "$OUT"

run_variant() {
	local name="$1"; shift
	local log="$OUT/$name.log"
	local store="$OUT/store-$name"
	mkdir -p "$store"

	echo "=== $name ==="
	( cd "$ROOT/server" && ./trellis2-server.exe \
		-lib "$ROOT/build/bin/Release/trellis2.dll" \
		-ggufs "$ROOT/ggufs" -store "$store" -addr ":$PORT" "$@" ) > "$log" 2>&1 &
	local pid=$!

	# The server answers /api/info once the library and models are up.
	local ready=0
	for _ in $(seq 1 240); do
		if curl -sf "http://127.0.0.1:$PORT/api/info" > /dev/null 2>&1; then ready=1; break; fi
		kill -0 "$pid" 2>/dev/null || break
		sleep 1
	done
	if [ "$ready" != "1" ]; then
		echo "  server did not come up; see $log"
		kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
		return 1
	fi

	local job
	job=$(curl -sf -X POST "http://127.0.0.1:$PORT/api/generate" \
		-F "image=@$IMAGE" -F "quality=$QUALITY" -F "seed=$SEED" \
		| sed -n 's/.*"job"[^"]*"\([^"]*\)".*/\1/p')
	if [ -z "$job" ]; then
		echo "  generate was not accepted; see $log"
		kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
		return 1
	fi
	echo "  job $job"

	local state=""
	for _ in $(seq 1 900); do
		state=$(curl -sf "http://127.0.0.1:$PORT/api/job/$job" | sed -n 's/.*"state"[^"]*"\([^"]*\)".*/\1/p')
		case "$state" in
			done|error|failed) break ;;
		esac
		kill -0 "$pid" 2>/dev/null || break
		sleep 2
	done
	echo "  state: ${state:-unknown}"

	kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

	grep -h '\[mesh\]' "$log" | tail -2 | sed 's/^/  /'
	grep -h 'finest-level expansion' "$log" | tail -1 | sed 's/^/  /'
}

G="$ROOT/ggufs"
run_variant f16
run_variant dec32 -shape-dec "$G/shape_dec_f32.gguf"
run_variant all32 \
	-dino "$G/dino_f32.gguf" \
	-flow "$G/ss_flow_f32.gguf" \
	-dec "$G/ss_dec_f32.gguf" \
	-slat "$G/slat_flow_f32.gguf" \
	-slat-hr "$G/slat_flow_1024_f32.gguf" \
	-shape-dec "$G/shape_dec_f32.gguf"

echo
echo "logs in $OUT"
