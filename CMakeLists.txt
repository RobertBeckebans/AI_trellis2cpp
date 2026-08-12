cmake_minimum_required(VERSION 3.14)
project(trellis2.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────────────────────────────────
# External ggml option
#
# When OFF (default): vendor ggml as a subdirectory — the upstream behavior.
# When ON : skip the bundled ggml and resolve `find_package(ggml CONFIG)`
#           against an installed ggml. Lets a downstream project link multiple
#           ggml-based libraries against a single shared ggml so they don't
#           ship duplicate copies into the same process.
# ─────────────────────────────────────────────────────────────────────────
option(TRELLIS2_USE_EXTERNAL_GGML "Consume external ggml via find_package() instead of the bundled submodule" OFF)

# libFuzzer harnesses (clang only). The sanitizer flags apply globally — set
# BEFORE any target so the whole tree (ggml, stb decode, preprocessing,
# loaders) is instrumented. Harness targets live in fuzz/ (added at the end).
option(TRELLIS2_FUZZ "Build libFuzzer harnesses (requires clang)" OFF)
set(TRELLIS2_FUZZ_SANITIZERS "address,undefined" CACHE STRING
    "Sanitizers for the fuzz build")
if(TRELLIS2_FUZZ)
    add_compile_options(-fsanitize=${TRELLIS2_FUZZ_SANITIZERS} -fsanitize=fuzzer-no-link -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=${TRELLIS2_FUZZ_SANITIZERS})
endif()

# ggml options — enable Metal on Apple (bundled-ggml path only)
option(TRELLIS2_METAL "Enable Metal backend" ON)
if(APPLE AND TRELLIS2_METAL AND NOT TRELLIS2_USE_EXTERNAL_GGML)
    set(GGML_METAL              ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

if(TRELLIS2_USE_EXTERNAL_GGML)
    find_package(ggml CONFIG REQUIRED)
    set(TRELLIS2_GGML_TARGET ggml::ggml)
else()
    add_subdirectory(ggml)
    set(TRELLIS2_GGML_TARGET ggml)
endif()

add_library(trellis2
    trellis2.cpp trellis2.h
    trellis2_capi.cpp trellis2_capi.h
    mesh_export.cpp mesh_export.h
    print_remesh.cpp print_remesh.h
    third_party/xatlas/xatlas.cpp)
target_include_directories(trellis2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)
# trellis2_capi.cpp reuses the example isosurface extractor for the demo path;
# mesh_export.cpp vendors xatlas for the optional chart-based UV unwrap.
target_include_directories(trellis2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xatlas
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinybvh)

# print_remesh.cpp traverses tinybvh's nodes for the closest-surface PBR
# projection, which is why the texture path no longer needs CGAL.
#
# tiny_bvh.h's implementation half uses std::scoped_lock in its build job
# system, so it needs C++17 — but the library itself stays C++14 (AGENTS.md
# reserves C++17 for the optional CGAL path). Isolating the one implementation
# TU in its own target keeps that promise: only this object is C++17, while
# print_remesh.cpp includes the declaration half as plain C++14.
#
# TINYBVH_NO_SIMD is deliberate, not a workaround. tiny_bvh.h assumes that a
# compiler defining _MSC_VER accepts SSE4.2/AVX intrinsics without target flags,
# which is true for MSVC but not for clang targeting the MSVC ABI — that
# combination fails to compile the wide-BVH ray traversal. The alternative,
# adding -mavx2 here, would raise libtrellis2's CPU baseline to Haswell for the
# whole library. We only use the scalar binned-SAH builder and our own scalar
# nearest-point traversal, so none of the SIMD ray paths are called anyway.
add_library(trellis2_tinybvh STATIC third_party/tinybvh/tiny_bvh_impl.cpp)
target_include_directories(trellis2_tinybvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinybvh)
target_compile_definitions(trellis2_tinybvh PUBLIC TINYBVH_NO_SIMD)
target_compile_features(trellis2_tinybvh PRIVATE cxx_std_17)
set_property(TARGET trellis2_tinybvh PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(trellis2_tinybvh PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(trellis2_tinybvh PRIVATE /w /bigobj)
else()
    target_compile_options(trellis2_tinybvh PRIVATE -w)
endif()
target_link_libraries(trellis2 PRIVATE trellis2_tinybvh)

# Select the tinybvh backend inside a CGAL build to compare the two in one binary.
option(TRELLIS2_FORCE_TINYBVH "Use the tinybvh projection backend even when CGAL is available" OFF)
if(TRELLIS2_FORCE_TINYBVH)
    target_compile_definitions(trellis2 PRIVATE TRELLIS2_FORCE_TINYBVH=1)
endif()

find_package(Threads REQUIRED)  # xatlas uses std::thread

if(TRELLIS2_USE_EXTERNAL_GGML AND DEFINED GGML_INCLUDE_DIR)
    target_include_directories(trellis2 PUBLIC ${GGML_INCLUDE_DIR})
endif()

target_link_libraries(trellis2 PUBLIC ${TRELLIS2_GGML_TARGET} PRIVATE Threads::Threads)
target_compile_features(trellis2 PUBLIC cxx_std_14)

# CGAL Alpha Wrap turns arbitrary TRELLIS triangle soups into watertight,
# intersection-free 2-manifolds for printing. Auto-detect it so ordinary builds
# retain no CGAL dependency; the C API reports whether the backend is present.
option(TRELLIS2_CGAL "Enable CGAL Alpha Wrap print remeshing when available" ON)
if(TRELLIS2_CGAL)
    find_package(CGAL 5.5 QUIET)
    if(CGAL_FOUND)
        target_link_libraries(trellis2 PRIVATE CGAL::CGAL)
        target_compile_definitions(trellis2 PRIVATE TRELLIS2_USE_CGAL=1)
        target_compile_features(trellis2 PRIVATE cxx_std_17)
        set(TRELLIS2_HAVE_CGAL ON)
        message(STATUS "CGAL Alpha Wrap print remeshing: enabled")
    else()
        message(STATUS "CGAL Alpha Wrap print remeshing: unavailable (CGAL >= 5.5 not found)")
    endif()
endif()

# AutoRemesher turns the dense dual-grid soup into quad-dominant mid-poly
# topology. The core is vendored under third_party/autoremesher (MIT), but it
# needs Eigen, which is MPL-2.0 and therefore consumed externally rather than
# vendored — `vcpkg install eigen3`, or let the vcpkg.json manifest resolve it.
# See THIRD_PARTY.md and docs/plan/autoremesher-quad-remesh.md.
#
# No version is passed to find_package on purpose: Eigen's own
# Eigen3ConfigVersion.cmake treats a single requested version as an *exact*
# component match (so `Eigen3 5.0` would reject 5.0.1), and a version range
# would require CMake 3.19 while this project targets 3.14. Check afterwards.
option(TRELLIS2_AUTOREMESHER "Enable AutoRemesher quad remeshing when Eigen 5.x is available" ON)
if(TRELLIS2_AUTOREMESHER)
    find_package(Eigen3 CONFIG QUIET)
    if(Eigen3_FOUND AND Eigen3_VERSION VERSION_GREATER_EQUAL 5.0
                    AND Eigen3_VERSION VERSION_LESS 6.0)
        set(TRELLIS2_AUTOREMESHER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/autoremesher)
        set(TRELLIS2_AUTOREMESHER_SOURCES
            ${TRELLIS2_AUTOREMESHER_DIR}/t2_ar_log.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/autoremesher.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/constrainedleastsquares.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/framefield.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/isotropicremesher.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/meshseparator.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/mixedintegerleastsquares.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/parameterizer.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/positionkey.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/quadextractor.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/quadparameterizer.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/singularitysimplifier.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/src/AutoRemesher/surfacemesh.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/isotropicremesher/axisalignedboundingboxtree.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/isotropicremesher/isotropichalfedgemesh.cpp
            ${TRELLIS2_AUTOREMESHER_DIR}/isotropicremesher/isotropicremesher.cpp)
        target_sources(trellis2 PRIVATE ${TRELLIS2_AUTOREMESHER_SOURCES})
        # compat/ resolves the upstream <tbb/...> includes onto our std::thread
        # shim, so no real TBB is required. See third_party/autoremesher/PATCHES.md.
        # Deliberately NOT the vendor root: on a case-insensitive filesystem it
        # would shadow the standard <version> header with the VERSION file.
        # t2_ar_log.h and t2_tbb_shim.h are reached by relative include anyway.
        target_include_directories(trellis2 PRIVATE
            ${TRELLIS2_AUTOREMESHER_DIR}/include
            ${TRELLIS2_AUTOREMESHER_DIR}/isotropicremesher
            ${TRELLIS2_AUTOREMESHER_DIR}/compat)
        target_link_libraries(trellis2 PRIVATE Eigen3::Eigen)
        target_compile_definitions(trellis2 PRIVATE TRELLIS2_USE_AUTOREMESHER=1)
        # Vendored code is not held to our warning level, and upstream's
        # autoremesher.pro sets these globally: _USE_MATH_DEFINES for M_PI on
        # the Windows CRT, NOMINMAX so <windows.h> does not break std::min/max.
        # /bigobj is an upstream requirement for the parameterizer objects.
        if(MSVC)
            set_source_files_properties(${TRELLIS2_AUTOREMESHER_SOURCES}
                PROPERTIES COMPILE_DEFINITIONS "_USE_MATH_DEFINES;NOMINMAX"
                           COMPILE_OPTIONS "/w;/bigobj")
        else()
            set_source_files_properties(${TRELLIS2_AUTOREMESHER_SOURCES}
                PROPERTIES COMPILE_DEFINITIONS "_USE_MATH_DEFINES;NOMINMAX"
                           COMPILE_OPTIONS "-w")
        endif()
        set(TRELLIS2_HAVE_AUTOREMESHER ON)
        message(STATUS "AutoRemesher quad remeshing: enabled (Eigen ${Eigen3_VERSION})")
    elseif(Eigen3_FOUND)
        message(STATUS "AutoRemesher quad remeshing: unavailable (Eigen ${Eigen3_VERSION} found, 5.x required)")
    else()
        message(STATUS "AutoRemesher quad remeshing: unavailable (Eigen 5.x not found; `vcpkg install eigen3`)")
    endif()
endif()

set_property(TARGET trellis2 PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(trellis2 PUBLIC  TRELLIS2_SHARED)
    target_compile_definitions(trellis2 PRIVATE TRELLIS2_BUILD)
endif()

option(TRELLIS2_BUILD_EXAMPLES "Build example executables" ON)
if(TRELLIS2_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

option(TRELLIS2_BUILD_TESTS "Build test executables" OFF)
if(TRELLIS2_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TRELLIS2_FUZZ)
    add_subdirectory(fuzz)
endif()
