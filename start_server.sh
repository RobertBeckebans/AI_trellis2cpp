#!/usr/bin/env bash
#
# --- Build what is stale, then start the Go demo server (cargo-run style) ---
#   Usage: ./start_server.sh [build dir]      (default: build)
#   e.g.   ./start_server.sh build-master
#
#   Both build steps are incremental and no-op when nothing changed, so this is
#   the normal way to start the server, not just a first-run helper.
#
#   The CMake build dir has to be configured once beforehand - that is where
#   the backend is chosen, which this script must not guess. Use
#   cmake-ninja-linux-clang.sh (Vulkan) or cmake-ninja-linux-rocm-cgal-quad.sh
#   (HIP), or configure by hand.
#
#   Unlike start_server.bat this needs no library search path: CMake stamps a
#   RUNPATH pointing at the output directory into libtrellis2.so, so the loader
#   resolves libggml{,-base,-cpu,-vulkan,-hip}.so.0 on its own. LD_LIBRARY_PATH
#   is exported anyway because that RUNPATH is an absolute path into the build
#   tree - it stops being true the moment the tree is moved or copied.

set -uo pipefail

# Everything is relative to the script's own directory, not the caller's, so
# this works from any shell.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD="${1:-build}"

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  cat >&2 <<EOF
$BUILD/ is not configured.
Configure and build it once, e.g.:
  ./cmake-ninja-linux-clang.sh            (Vulkan)
  ./cmake-ninja-linux-rocm-cgal-quad.sh   (HIP + CGAL + quad stage)
EOF
  exit 1
fi

command -v go >/dev/null 2>&1 || {
  echo "go not found - the server is a Go program. sudo apt install golang-go" >&2
  exit 1
}

echo "[1/2] libtrellis2.so"
# --config is ignored by single-config generators, so this covers both the
# Ninja Multi-Config builds the cmake-ninja-linux-*.sh scripts produce and a
# plain -DCMAKE_BUILD_TYPE=Release tree.
if ! cmake --build "$BUILD" --config Release; then
  cat >&2 <<EOF

Library build failed - see the output above. A common cause:
  * "CMake Generate step failed" after a ggml bump: stale cache entries.
    Drop them individually, e.g. for MATH_LIBRARY-NOTFOUND:
      cmake -B $BUILD -U MATH_LIBRARY
    Otherwise reconfigure $BUILD/ (cmake-ninja-linux-*.sh).
EOF
  exit 1
fi

# Multi-config puts the library under bin/<config>/, single-config directly in
# the build dir. Find it rather than making the caller care.
LIB=""
for candidate in "$BUILD/bin/Release/libtrellis2.so" \
                 "$BUILD/bin/libtrellis2.so" \
                 "$BUILD/libtrellis2.so"; do
  if [ -f "$candidate" ]; then
    LIB="$(cd "$(dirname "$candidate")" && pwd)/libtrellis2.so"
    break
  fi
done
if [ -z "$LIB" ]; then
  echo "not found: libtrellis2.so under $BUILD/. Was the build configured with" >&2
  echo "-DBUILD_SHARED_LIBS=ON? The static default builds no shared library." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[2/2] trellis2-server"
cd server
if ! go build -o trellis2-server .; then
  echo >&2
  echo "Server build failed." >&2
  exit 1
fi

# GGML_CUDA_DISABLE_GRAPHS is deliberately not set here: the server sets it
# itself (see the comment in server/main.go).

echo
./trellis2-server \
  -lib "$LIB" \
  -ggufs ../ggufs \
  -store ../generations \
  -addr :8742 \
  -unload-idle

RC=$?
if [ "$RC" != "0" ]; then
  echo >&2
  echo "Server exited with an error (exit $RC)." >&2
fi
exit "$RC"
