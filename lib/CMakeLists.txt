list(APPEND lib_alluvion_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/store.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_device_pointer.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_pinned_pointer.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_graphical_resource.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_mesh_buffer.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_texture.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_vao.cu
  ${CMAKE_CURRENT_LIST_DIR}/unique_vbo.cu
  ${CMAKE_CURRENT_LIST_DIR}/mesh_buffer.cu
  ${CMAKE_CURRENT_LIST_DIR}/mesh.cu
  ${CMAKE_CURRENT_LIST_DIR}/allocator.cu
  ${CMAKE_CURRENT_LIST_DIR}/graphical_allocator.cu
  ${CMAKE_CURRENT_LIST_DIR}/constants.cu
  ${CMAKE_CURRENT_LIST_DIR}/display.cu
  ${CMAKE_CURRENT_LIST_DIR}/shading_program.cu
  ${CMAKE_CURRENT_LIST_DIR}/complete_framebuffer.cu
  ${CMAKE_CURRENT_LIST_DIR}/camera.cu
  ${CMAKE_CURRENT_LIST_DIR}/trackball.cu
  ${CMAKE_CURRENT_LIST_DIR}/glad.c
)

if (ALLUVION_BUILD_EXECUTABLES OR ALLUVION_BUILD_TESTS)
  add_library(lib_alluvion
    ${lib_alluvion_SOURCES}
  )

  target_link_libraries(lib_alluvion
    PRIVATE
      glfw
    PUBLIC
      Eigen3::Eigen
      OpenMP::OpenMP_CXX
  )


  target_include_directories(lib_alluvion
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  )
endif()

# The reason for compiling all source files
# https://forums.developer.nvidia.com/t/issue-with-cudamemcpytosymbol-and-separable-compilation/70850/11
# You can’t use device linking across a shared library interface.
# You can use relocatable device code in a shared library as
# long as the linking is entirely within the shared library.

# pybind11_add_module() uses LTO by default but it is not for CUDA
# How to disable LTO:
# https://stackoverflow.com/a/51933203
add_library(_alluvion
  MODULE
    ${lib_alluvion_SOURCES}
    pyalluvion.cu
)

set_target_properties(_alluvion
  PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_include_directories(_alluvion
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_compile_options(_alluvion
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(_alluvion
  PUBLIC
    glfw
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    pybind11::module
)
