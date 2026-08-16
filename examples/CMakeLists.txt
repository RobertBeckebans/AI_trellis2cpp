# Example: inspect a .dinodata DINOv3 conditioning file.
add_executable(dino_info dino_info.cpp)
target_link_libraries(dino_info PRIVATE trellis2)
target_compile_features(dino_info PRIVATE cxx_std_14)

# Example: inspect a converted SS-flow DiT GGUF (hparams + tensor inventory).
add_executable(ss_flow_info ss_flow_info.cpp)
target_link_libraries(ss_flow_info PRIVATE trellis2)
target_compile_features(ss_flow_info PRIVATE cxx_std_14)

# Example: stage-1 sampling, .dinodata cond -> sparse-structure latent z_s.
add_executable(ss_sample ss_sample.cpp)
target_link_libraries(ss_sample PRIVATE trellis2)
target_compile_features(ss_sample PRIVATE cxx_std_14)

# Example: stage-1 decoding, z_s latent -> occupancy logit grid (64^3).
add_executable(ss_decode ss_decode.cpp)
target_link_libraries(ss_decode PRIVATE trellis2)
target_compile_features(ss_decode PRIVATE cxx_std_14)

# Example: decode + extract the occupancy isosurface as an OBJ (marching cubes).
add_executable(ss_mesh ss_mesh.cpp)
target_link_libraries(ss_mesh PRIVATE trellis2)
target_compile_features(ss_mesh PRIVATE cxx_std_14)

# Example: image -> preprocessed 512x512 -> DINOv3 conditioning (.dinodata).
add_executable(dino_encode dino_encode.cpp)
target_link_libraries(dino_encode PRIVATE trellis2)
target_compile_features(dino_encode PRIVATE cxx_std_14)

# Example: export a demo mesh (T2MESH01/02/03) into a portable GLB (no GPU).
add_executable(mesh2glb mesh2glb.cpp)
target_link_libraries(mesh2glb PRIVATE trellis2)
target_compile_features(mesh2glb PRIVATE cxx_std_14)

# Mesh a dumped dual-grid decoder output (.fdgvox) into a T2MESH01 blob. Lets
# the PyTorch reference's voxels go through our extractor, so a reference mesh
# can be compared against the backends without a second mesher in the picture.
# Header-only (examples/flexible_dual_grid.h), so no model and no GPU.
add_executable(dual_grid_cli dual_grid_cli.cpp)
target_include_directories(dual_grid_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(dual_grid_cli PRIVATE cxx_std_14)

# Smoke/benchmark tool for the quad stage: OBJ or .t2mesh in, quad OBJ out.
# Drives t2quad::remesh, i.e. the same path the server uses including input
# preparation - talking to the vendored core directly would skip exactly the
# part most likely to be responsible for a bad result.
if(TRELLIS2_HAVE_AUTOREMESHER)
    add_executable(quad_remesh_cli quad_remesh_cli.cpp)
    target_link_libraries(quad_remesh_cli PRIVATE trellis2)
    target_compile_features(quad_remesh_cli PRIVATE cxx_std_14)
    if(WIN32)
        target_link_libraries(quad_remesh_cli PRIVATE psapi)
    endif()
endif()
