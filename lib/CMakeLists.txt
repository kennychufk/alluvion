add_library(lib_alluvion SHARED
  store.cu
  unique_device_pointer.cu
  unique_graphical_resource.cu
  allocator.cu
  graphical_allocator.cu
  constants.cu
  display.cu
  shading_program.cu
  camera.cu
  trackball.cu
  glad.c
)

target_link_libraries(lib_alluvion
  PRIVATE
    glfw
)


target_include_directories(lib_alluvion
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

set_target_properties(lib_alluvion
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES OFF
)
