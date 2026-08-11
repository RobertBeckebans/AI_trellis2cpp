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
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xatlas)

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
