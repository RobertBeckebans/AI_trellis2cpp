#!/usr/bin/env bash
#
# --- Shared-library build with BOTH optional backends, for comparing them ---
#   Linux port of cmake-ninja-win64-rocm-cgal-quad.bat: the CGAL Alpha Wrap
#   print wrap *and* the AutoRemesher quad stage in one library, so the viewer
#   offers "watertight print wrap (CGAL)" and "quad retopology" at the same
#   time and they can be judged against each other on one model.
#
#   The interesting third case is both together: Alpha Wrap first hands the
#   remesher a closed 2-manifold, which is the input AutoRemesher expects.
#   On raw dual-grid geometry it produces thousands of open edges; on wrapped
#   geometry it should not. That combination is still NOT guaranteed
#   watertight - the quad stage can reopen boundaries - so check the
#   "boundary edges" figure the viewer reports before trusting it.
#
# What differs from the Windows script, and why:
#
#   * No vcpkg by default. CGAL comes from the distribution and Eigen 5.x from
#     a local prefix this script can install for you (--install-eigen). Set
#     VCPKG_ROOT to fall back to the manifest path the .bat uses.
#   * ROCm has to be a *current* SDK. Details in the ROCm block below - the
#     short version is that neither the distribution packages nor the runtimes
#     bundled with PyTorch or ollama can build ggml-hip for gfx1201.
#   * Shared libraries land in the LIBRARY output directory on Linux, so both
#     RUNTIME and LIBRARY are pointed at build/bin - that keeps
#     libtrellis2.so next to the example binaries, as the .bat does for DLLs.
#
# Overridable from the environment:
#   ROCM_ROOT       ROCm SDK prefix            (default: autodetected)
#   GPU_TARGET      AMDGPU arch                (default: gfx1201)
#   EIGEN5_ROOT     prefix holding Eigen 5.x   (default: $HOME/.local/eigen5)
#   EIGEN5_VERSION  tag to install             (default: 5.0.1)
#   VCPKG_ROOT      if set, resolve Eigen+CGAL through vcpkg instead
#   FORCE_TINYBVH   ON/OFF, projection backend (default: ON)
#   BUILD_CONFIG    Release/Debug/RelWithDebInfo (default: Release)
#
# Arguments:
#   --install-eigen   install Eigen $EIGEN5_VERSION into $EIGEN5_ROOT first

set -euo pipefail

GPU_TARGET="${GPU_TARGET:-gfx1201}"
BUILD_CONFIG="${BUILD_CONFIG:-Release}"
EIGEN5_ROOT="${EIGEN5_ROOT:-$HOME/.local/eigen5}"
EIGEN5_VERSION="${EIGEN5_VERSION:-5.0.1}"

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
  # Escape hatch: the same manifest path the Windows script uses. The cgal
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

# --- ROCm SDK ---------------------------------------------------------------
#
# Three things on a machine can look like ROCm and only one of them builds:
#
#   * A ROCm *runtime*: the .so files inside a PyTorch ROCm wheel (torch/lib)
#     or in /usr/local/lib/ollama/rocm. Current, gfx1201-capable, and useless
#     here - no headers, no hipblas/rocblas CMake package.
#   * The *distribution* packages. Ubuntu 25.10 does carry an SDK (hipcc,
#     libamdhip64-dev, libhipblas-dev, librocblas-dev), but it is ROCm 5.5/5.7
#     era: /usr/lib/x86_64-linux-gnu/rocblas/2.47.0/library ships Tensile
#     kernels for gfx803 ... gfx1102 only. Nothing for gfx1201, so a build
#     against it would configure and then have no kernels for this GPU.
#   * A current SDK from AMD, which is what this script wants.

if [ -z "${ROCM_ROOT:-}" ]; then
  for candidate in "${ROCM_PATH:-}" /opt/rocm $(ls -d /opt/rocm-* 2>/dev/null | sort -V -r); do
    if [ -n "$candidate" ] && [ -x "$candidate/llvm/bin/clang++" ]; then
      ROCM_ROOT="$candidate"
      break
    fi
  done
fi

# TheRock ships the same SDK as Python wheels, which is a no-root route and
# fits the uv setup in ~/Projects/Python_UV_CUDA_ROCM_Template. Activate that
# environment before running this script and it gets picked up here.
if [ -z "${ROCM_ROOT:-}" ] && command -v rocm-sdk >/dev/null 2>&1; then
  ROCM_ROOT="$(rocm-sdk path --root 2>/dev/null || true)"
fi

if [ -z "${ROCM_ROOT:-}" ]; then
  die "No ROCm SDK found. Set ROCM_ROOT, or install one - the distribution
packages are not an option for ${GPU_TARGET}, see the comment above:

  wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/\$(lsb_release -cs)/amdgpu-install_all.deb
  sudo apt install ./amdgpu-install_all.deb
  sudo amdgpu-install --usecase=rocm

or, without root, into a virtual environment:

  uv pip install 'rocm[libraries,devel]'

Alternatively build without HIP: -DGGML_HIP=OFF -DGGML_VULKAN=ON."
fi

ROCM_BIN="$ROCM_ROOT/llvm/bin"
export PATH="$ROCM_ROOT/bin:$ROCM_BIN:$PATH"

HIPBLAS_DIR="$ROCM_ROOT/lib/cmake/hipblas"
[ -d "$HIPBLAS_DIR" ] || die "ROCm at $ROCM_ROOT has no hipblas CMake package ($HIPBLAS_DIR).
Install the hipBLAS/rocBLAS development packages (amdgpu-install --usecase=rocm)."

# rocBLAS carries precompiled kernels per architecture; a missing one only
# surfaces at run time as a solution-not-found error, so check it here.
if ! ls "$ROCM_ROOT"/lib/rocblas/library/*"$GPU_TARGET"* >/dev/null 2>&1; then
  echo "WARNING: rocBLAS at $ROCM_ROOT ships no kernels for $GPU_TARGET - the build will"
  echo "         succeed but fail at run time. Check the ROCm version."
fi

rm -rf build

cmake -B build \
  -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER="$ROCM_BIN/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_BIN/clang++" \
  "${TOOLCHAIN_ARGS[@]}" \
  ${PREFIX_PATH:+-DCMAKE_PREFIX_PATH="$PREFIX_PATH"} \
  -DGGML_VULKAN=OFF \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$GPU_TARGET" \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN_OUT" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$BIN_OUT" \
  -DTRELLIS2_CGAL=ON \
  -DTRELLIS2_AUTOREMESHER=ON \
  -DTRELLIS2_FORCE_TINYBVH="$FORCE_TINYBVH" \
  -DTRELLIS2_BUILD_EXAMPLES=ON \
  -DTRELLIS2_BUILD_TESTS=ON \
  -Dhipblas_DIR="$HIPBLAS_DIR" \
  . || die "CMake configure failed."

# The configure output above must contain all three:
#   -- CGAL Alpha Wrap print remeshing: enabled
#   -- PBR projection backend: cgal            (or tinybvh)
#   -- AutoRemesher quad remeshing: enabled (Eigen 5.x.x)
# If CGAL says "unavailable", libcgal-dev is missing and the print wrap will be
# missing from the viewer.

# No -j: ninja already schedules nproc+2 jobs. Pass one explicitly only to cap it.
if ! cmake --build build --config "$BUILD_CONFIG"; then
  die "Build failed.

If it breaks in print_remesh.cpp with
  'boost/mpl/aux_/preprocessed/plain / or.hpp' file not found
then that is the known Boost.MPL/clang problem - it happens with unmodified
sources too and has nothing to do with this change. See
docs/progress/autoremesher-quad-remesh_phase2-tinybvh-projection.md"
fi

cat <<EOF

Done: build/bin/$BUILD_CONFIG/libtrellis2.so  (+ libggml.so, libggml-base.so, libggml-cpu.so, libggml-hip.so)

Comparison in the viewer: "watertight print wrap (CGAL)" and "quad retopology"
can be toggled separately and together. For every export the server logs a
timing block and reports quads / open edges / area preservation.

Offline on an already generated mesh, without the server:
  build/bin/$BUILD_CONFIG/quad_remesh_cli generations/<job>/mesh.t2mesh out.obj --target-quads 20000 --min-area 0
EOF
