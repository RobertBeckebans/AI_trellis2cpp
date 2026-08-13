# Numerical validation of the SS-flow DiT forward pass against a PyTorch
# reference (generate the reference with tests/ref_ss_flow.py).
add_executable(test_ss_flow_forward test_ss_flow_forward.cpp)
target_link_libraries(test_ss_flow_forward PRIVATE trellis2)
target_compile_features(test_ss_flow_forward PRIVATE cxx_std_14)

# Full flow-Euler sampling loop vs the PyTorch reference (tests/ref_ss_sample.py).
add_executable(test_ss_sample test_ss_sample.cpp)
target_link_libraries(test_ss_sample PRIVATE trellis2)
target_compile_features(test_ss_sample PRIVATE cxx_std_14)

# SS decoder forward vs the PyTorch reference (tests/ref_ss_dec.py).
add_executable(test_ss_dec test_ss_dec.cpp)
target_link_libraries(test_ss_dec PRIVATE trellis2)
target_compile_features(test_ss_dec PRIVATE cxx_std_14)

# PIL-compatible preprocessing byte-exactness vs the Python reference.
add_executable(test_preprocess test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE trellis2)
target_include_directories(test_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb)
target_compile_features(test_preprocess PRIVATE cxx_std_14)

# Deterministic border-connected black/white background removal (no models).
add_executable(test_background_removal test_background_removal.cpp)
target_link_libraries(test_background_removal PRIVATE trellis2)
target_compile_features(test_background_removal PRIVATE cxx_std_14)

# DINOv3 encoder per-layer parity vs the PyTorch reference
# (scripts/dump_dino_reference.py -> dumps/reference_dino.gguf).
add_executable(test_dino test_dino.cpp)
target_link_libraries(test_dino PRIVATE trellis2)
target_compile_features(test_dino PRIVATE cxx_std_14)

# Shape-SLAT flow + FDG decoder parity vs the PyTorch reference
# (scripts/dump_slat_reference.py -> dumps/reference_slat.gguf).
add_executable(test_slat test_slat.cpp)
target_link_libraries(test_slat PRIVATE trellis2)
target_compile_features(test_slat PRIVATE cxx_std_14)

# Cascade HR-stage parity vs the PyTorch reference
# (scripts/dump_cascade_reference.py -> dumps/reference_cascade.gguf).
add_executable(test_cascade test_cascade.cpp)
target_link_libraries(test_cascade PRIVATE trellis2)
target_compile_features(test_cascade PRIVATE cxx_std_14)

# The cascade HR token budget (src/cascade_tokens.h) on synthetic coordinate
# sets: the quantization formula, the step-down loop, and the 1024 floor.
# Header-only, so it needs no library and no model assets.
add_executable(test_cascade_tokens test_cascade_tokens.cpp)
target_include_directories(test_cascade_tokens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(test_cascade_tokens PRIVATE cxx_std_14)

# PBR-texture stage parity vs the PyTorch reference
# (scripts/dump_texture_reference.py -> dumps/reference_texture.gguf).
add_executable(test_texture test_texture.cpp)
target_link_libraries(test_texture PRIVATE trellis2)
target_compile_features(test_texture PRIVATE cxx_std_14)

# Self-contained isosurface extractor (examples/marching_cubes.h) on analytic
# fields — watertight-manifold + Euler-characteristic invariants. No model.
add_executable(test_marching_cubes test_marching_cubes.cpp)
target_include_directories(test_marching_cubes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
target_compile_features(test_marching_cubes PRIVATE cxx_std_14)

# Sparse PBR-volume trilinear sampling used by the integrated texture path.
add_executable(test_pbr_sampling test_pbr_sampling.cpp)
target_include_directories(test_pbr_sampling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(test_pbr_sampling PRIVATE cxx_std_14)

# CPU GLB bake smoke/regression test, including six-channel alpha preservation.
add_executable(test_mesh_export test_mesh_export.cpp)
target_link_libraries(test_mesh_export PRIVATE trellis2)
target_compile_features(test_mesh_export PRIVATE cxx_std_14)

# GPU-vs-CPU agreement of the finest decoder level's ops past 2^21 rows, where
# the 1024^3 cascade's output turns to garbage on ROCm/HIP. No model assets.
add_executable(test_large_rows test_large_rows.cpp)
target_link_libraries(test_large_rows PRIVATE trellis2)
target_compile_features(test_large_rows PRIVATE cxx_std_14)

# The block-wise finest level must match the single-graph one voxel for voxel.
add_executable(test_chunked_decode test_chunked_decode.cpp)
target_link_libraries(test_chunked_decode PRIVATE trellis2)
target_compile_features(test_chunked_decode PRIVATE cxx_std_14)

# Builds unconditionally since the tinybvh projection backend landed: the
# closest-surface half of print_remesh no longer needs CGAL, and the test now
# covers it in every build. The Alpha Wrap half reports itself as skipped from
# inside the test when CGAL is absent.
add_executable(test_print_remesh test_print_remesh.cpp)
target_link_libraries(test_print_remesh PRIVATE trellis2)
target_compile_features(test_print_remesh PRIVATE cxx_std_14)

# Structural invariants of the AutoRemesher quad stage on synthetic fixtures.
# Builds unconditionally — t2quad::available() is compiled either way — and the
# test skips itself (77) when the backend or Eigen 5.x was absent at configure.
add_executable(test_quad_remesh test_quad_remesh.cpp)
target_link_libraries(test_quad_remesh PRIVATE trellis2)
target_compile_features(test_quad_remesh PRIVATE cxx_std_14)

# ── ctest registration ────────────────────────────────────────────────────────
# Asset paths default to the in-repo locations produced by
# scripts/download_models.sh + converters + scripts/refgen.sh, overridable via
# TRELLIS2_GGUF_DIR / TRELLIS2_DUMPS. Tests exit 77 when assets are absent, so
# a fresh checkout still gets a green (skipped) suite.
set(T2_GGUFS "$ENV{TRELLIS2_GGUF_DIR}")
if(NOT T2_GGUFS)
    set(T2_GGUFS "${CMAKE_CURRENT_SOURCE_DIR}/../ggufs")
endif()
set(T2_DUMPS "$ENV{TRELLIS2_DUMPS}")
if(NOT T2_DUMPS)
    set(T2_DUMPS "${CMAKE_CURRENT_SOURCE_DIR}/../dumps")
endif()
set(T2_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME marching_cubes COMMAND test_marching_cubes)
add_test(NAME pbr_sampling COMMAND test_pbr_sampling)
add_test(NAME mesh_export COMMAND test_mesh_export)
add_test(NAME print_remesh COMMAND test_print_remesh)
add_test(NAME background_removal COMMAND test_background_removal)
add_test(NAME cascade_tokens COMMAND test_cascade_tokens)

# Skips (77) in a build without the AutoRemesher backend. No model assets.
add_test(NAME quad_remesh COMMAND test_quad_remesh)
set_tests_properties(quad_remesh PROPERTIES SKIP_RETURN_CODE 77)

# Skips (77) when no GPU backend is compiled in; slow because it moves ~1 GB.
add_test(NAME large_rows COMMAND test_large_rows)
set_tests_properties(large_rows PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;slow")

add_test(NAME chunked_decode COMMAND test_chunked_decode "${T2_GGUFS}/tex_dec_f16.gguf")
set_tests_properties(chunked_decode PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME preprocess COMMAND test_preprocess
         "${T2_DUMPS}/fixture_rgba.png" "${T2_DUMPS}/fixture_512.png")
set_tests_properties(preprocess PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME dino COMMAND test_dino
         "${T2_GGUFS}/dino_f32.gguf" "${T2_DUMPS}/reference_dino.gguf")
set_tests_properties(dino PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME ss_flow_forward COMMAND test_ss_flow_forward
         "${T2_GGUFS}/ss_flow_f32.gguf" "${T2_TESTS}/ss_flow_ref.bin")
set_tests_properties(ss_flow_forward PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME ss_sample COMMAND test_ss_sample
         "${T2_GGUFS}/ss_flow_f32.gguf" "${T2_TESTS}/ss_sample_ref.bin")
set_tests_properties(ss_sample PROPERTIES SKIP_RETURN_CODE 77 LABELS "model;slow")

add_test(NAME ss_dec COMMAND test_ss_dec
         "${T2_GGUFS}/ss_dec_f32.gguf" "${T2_TESTS}/ss_dec_ref.bin")
set_tests_properties(ss_dec PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME slat COMMAND test_slat
         "${T2_GGUFS}/slat_flow_f32.gguf" "${T2_GGUFS}/shape_dec_f32.gguf"
         "${T2_DUMPS}/reference_slat.gguf")
set_tests_properties(slat PROPERTIES SKIP_RETURN_CODE 77 LABELS "model;slow"
                     ENVIRONMENT "TRELLIS2_DINODATA=${T2_DUMPS}/fixture.dinodata")

add_test(NAME cascade COMMAND test_cascade
         "${T2_GGUFS}/slat_flow_f32.gguf" "${T2_GGUFS}/slat_flow_1024_f32.gguf"
         "${T2_GGUFS}/shape_dec_f32.gguf" "${T2_DUMPS}/reference_cascade.gguf")
set_tests_properties(cascade PROPERTIES SKIP_RETURN_CODE 77 LABELS "model;slow")

add_test(NAME texture COMMAND test_texture
         "${T2_GGUFS}/shape_enc_f16.gguf" "${T2_GGUFS}/tex_slat_flow_512_f16.gguf"
         "${T2_GGUFS}/tex_dec_f16.gguf" "${T2_DUMPS}/reference_texture.gguf")
set_tests_properties(texture PROPERTIES SKIP_RETURN_CODE 77 LABELS "model;slow")
