#!/usr/bin/env bash
#
# --- Vulkan shared-library build with both optional backends ---
#   The Linux counterpart to cmake-ninja-win64-vulkan.bat, but carrying the
#   same optional stages as cmake-ninja-linux-rocm.sh: CGAL Alpha
#   Wrap and the AutoRemesher quad stage. Diff the two scripts and the only
#   real difference is the compute backend.
#
#   Vulkan is the third data point next to CPU and ROCm when a stage comes out
#   wrong on one backend only (e.g. the 1024^3 shape decode truncating at 2^21
#   voxels on HIP, which the CPU path does not do). It is also the backend that
#   needs no ROCm SDK, so it builds on a plain Kubuntu with the host clang.
#
# Prerequisites, none of which need a Vulkan SDK tarball on Linux:
#   sudo apt install clang libvulkan-dev glslc spirv-headers libcgal-dev
#   ./cmake-ninja-linux-clang.sh --install-eigen     (once, for the quad stage)
#
# Overridable from the environment:
#   CC / CXX        host compiler              (default: clang / clang++)
#   EIGEN5_ROOT     prefix holding Eigen 5.x   (default: $HOME/.local/eigen5)
#   EIGEN5_VERSION  tag to install             (default: 5.0.1)
#   VCPKG_ROOT      if set, resolve Eigen+CGAL through vcpkg instead
#   FORCE_TINYBVH   ON/OFF, projection backend (default: ON)
#   BUILD_CONFIG    Release/Debug/RelWithDebInfo (default: Release)
#
# Arguments:
#   --install-eigen   install Eigen $EIGEN5_VERSION into $EIGEN5_ROOT first

set -euo pipefail

BUILD_CONFIG="${BUILD_CONFIG:-Release}"
EIGEN5_ROOT="${EIGEN5_ROOT:-$HOME/.local/eigen5}"
EIGEN5_VERSION="${EIGEN5_VERSION:-5.0.1}"
CC="${CC:-clang}"
CXX="${CXX:-clang++}"

# Which library does the closest-surface search for the texture rebake.
# OFF = CGAL (the reference), ON = tinybvh. Flip it to compare the two
# projection backends; the configure output prints which one is active.
FORCE_TINYBVH="${FORCE_TINYBVH:-ON}"

INSTALL_EIGEN=0
for arg in "$@"; do
  case "$arg" in
    --install-eigen) INSTALL_EIGEN=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC_DIR"

# CMAKE_RUNTIME/LIBRARY_OUTPUT_DIRECTORY must be absolute (relative paths get
# reinterpreted per-subdir by ggml's nested CMakeLists).
BIN_OUT="$SRC_DIR/build/bin"

die() { printf '\n%s\n' "$*" >&2; exit 1; }

# --- Eigen 5.x and CGAL -----------------------------------------------------
#
# Eigen: the quad stage requires 5.x, and Debian/Ubuntu ship libeigen3-dev
# 3.4.0, which the version check in CMakeLists.txt rejects. Upstream tags 5.0.0
# and 5.0.1 exist and Eigen is header-only, so it installs into a user prefix
# without root - that is what --install-eigen does. BLAS/LAPACK have to be
# switched off, otherwise `cmake --install` looks for a libeigen_blas_static.a
# that a headers-only run never built.
#
# CGAL: apt install libcgal-dev (Ubuntu 25.10: 6.0.1). CGAL's own
# CGALConfigVersion.cmake treats every newer version as compatible, so the
# find_package(CGAL 5.5) in CMakeLists.txt accepts 6.0.1. Note that CGALConfig
# pulls in GMP/MPFR/Boost with find_package(... REQUIRED) - if those are
# missing the configure aborts outright instead of quietly reporting CGAL as
# unavailable. Installing libcgal-dev through apt drags them in.

install_eigen() {
  local src="${TMPDIR:-/tmp}/eigen-$EIGEN5_VERSION"
  echo "Installing Eigen $EIGEN5_VERSION into $EIGEN5_ROOT ..."
  rm -rf "$src"
  git clone --depth 1 --branch "$EIGEN5_VERSION" https://gitlab.com/libeigen/eigen.git "$src"
  cmake -S "$src" -B "$src/build" \
    -DCMAKE_INSTALL_PREFIX="$EIGEN5_ROOT" \
    -DEIGEN_BUILD_TESTING=OFF \
    -DEIGEN_BUILD_BLAS=OFF \
    -DEIGEN_BUILD_LAPACK=OFF \
    -DEIGEN_BUILD_DOC=OFF
  cmake --install "$src/build"
  rm -rf "$src"
}

PREFIX_PATH=""
TOOLCHAIN_ARGS=()

if [ -n "${VCPKG_ROOT:-}" ]; then
  # Escape hatch: the same manifest path the Windows scripts use. The cgal
  # feature is opt-in so a default configure never resolves a GPL package.
  [ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ] ||
    die "VCPKG_ROOT=$VCPKG_ROOT has no scripts/buildsystems/vcpkg.cmake."
  echo "Resolving Eigen and CGAL through vcpkg at $VCPKG_ROOT."
  TOOLCHAIN_ARGS=(
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
    -DVCPKG_MANIFEST_FEATURES=cgal
  )
else
  if [ "$INSTALL_EIGEN" = "1" ]; then
    install_eigen
  fi

  if [ -d "$EIGEN5_ROOT" ]; then
    PREFIX_PATH="$EIGEN5_ROOT"
  else
    echo "WARNING: no Eigen 5.x prefix at $EIGEN5_ROOT - the quad stage will configure as"
    echo "         unavailable. Run this script once with --install-eigen."
  fi

  if ! [ -d /usr/include/CGAL ] && [ -z "${CGAL_DIR:-}" ]; then
    echo "WARNING: no CGAL headers in /usr/include/CGAL - the print wrap will be missing"
    echo "         from the viewer. Install it with: sudo apt install libcgal-dev"
  fi
fi

# --- Vulkan -----------------------------------------------------------------
#
# ggml-vulkan compiles its compute shaders at build time, so a loader alone is
# not enough: it wants find_package(Vulkan COMPONENTS glslc REQUIRED) plus
# find_package(SPIRV-Headers CONFIG REQUIRED). On Linux all of it comes from
# the distribution; VULKAN_SDK is a Windows concern and is not needed here.
# CMake will also report glslangValidator as a missing Vulkan component - that
# one is optional and does not stop the build.

command -v "$CC" >/dev/null 2>&1 || die "$CC not found. sudo apt install clang"

# ggml-vulkan.cpp includes both the C and the C++ header.
[ -f /usr/include/vulkan/vulkan.hpp ] ||
  die "No Vulkan headers. sudo apt install libvulkan-dev"

command -v glslc >/dev/null 2>&1 ||
  die "glslc not found - ggml-vulkan compiles its shaders with it and CMake
requires it as a Vulkan component. sudo apt install glslc"

# SPIRV-Headers is a config-mode package, so a plain header check would miss
# it; look for the config file the way find_package does.
if [ -z "$(ls -d /usr/share/cmake/SPIRV-Headers \
                /usr/local/share/cmake/SPIRV-Headers \
                /usr/lib/*/cmake/SPIRV-Headers 2>/dev/null)" ]; then
  die "SPIRV-Headers not found - ggml-vulkan requires it in config mode.
sudo apt install spirv-headers"
fi

rm -rf build

cmake -B build \
  -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  "${TOOLCHAIN_ARGS[@]}" \
  ${PREFIX_PATH:+-DCMAKE_PREFIX_PATH="$PREFIX_PATH"} \
  -DGGML_VULKAN=ON \
  -DGGML_HIP=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN_OUT" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$BIN_OUT" \
  -DTRELLIS2_CGAL=ON \
  -DTRELLIS2_AUTOREMESHER=ON \
  -DTRELLIS2_FORCE_TINYBVH="$FORCE_TINYBVH" \
  -DTRELLIS2_BUILD_EXAMPLES=ON \
  -DTRELLIS2_BUILD_TESTS=ON \
  . || die "CMake configure failed."

# The configure output above must contain all three:
#   -- CGAL Alpha Wrap print remeshing: enabled
#   -- PBR projection backend: cgal            (or tinybvh)
#   -- AutoRemesher quad remeshing: enabled (Eigen 5.x.x)

# No -j: ninja already schedules nproc+2 jobs. Pass one explicitly only to cap
# it - the generated ggml-vulkan shader units are large enough (mul_mm.comp.cpp
# is ~115 MB) that a memory-tight machine may want fewer.
cmake --build build --config "$BUILD_CONFIG" || die "Build failed."

cat <<EOF

Done: build/bin/$BUILD_CONFIG/libtrellis2.so  (+ libggml.so, libggml-base.so, libggml-cpu.so, libggml-vulkan.so)

Start from server/:
  ./trellis2-server -lib ../build/bin/$BUILD_CONFIG/libtrellis2.so

Offline on an already generated mesh, without the server:
  build/bin/$BUILD_CONFIG/quad_remesh_cli generations/<job>/mesh.t2mesh out.obj --target-quads 20000 --min-area 0
EOF
