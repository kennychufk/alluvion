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
  ${CMAKE_CURRENT_LIST_DIR}/pile.cu
  ${CMAKE_CURRENT_LIST_DIR}/constants.cu
  ${CMAKE_CURRENT_LIST_DIR}/display.cu
  ${CMAKE_CURRENT_LIST_DIR}/shading_program.cu
  ${CMAKE_CURRENT_LIST_DIR}/typesetter.cu
  ${CMAKE_CURRENT_LIST_DIR}/camera.cu
  ${CMAKE_CURRENT_LIST_DIR}/trackball.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/bounding_sphere_hierarchy.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/triangle_mesh.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/entity_iterators.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/entity_containers.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/sphere_distance.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/infinite_cylinder_distance.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/mesh_distance.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/point_triangle_distance.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/discrete_grid.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/cubic_lagrange_discrete_grid.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/volume_integration.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/sph_kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/dg/gauss_quadrature.cu
  ${CMAKE_CURRENT_LIST_DIR}/glad.c
)

add_library(lib_alluvion
  ${lib_alluvion_SOURCES}
)

target_link_libraries(lib_alluvion
  PRIVATE
    glfw
    ${FREETYPE_LIBRARIES}
  PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)


target_include_directories(lib_alluvion
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    ${FREETYPE_INCLUDE_DIRS}
)

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
    ${FREETYPE_INCLUDE_DIRS}
)

target_link_libraries(_alluvion
  PUBLIC
    glfw
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    pybind11::module
  PRIVATE
    ${FREETYPE_LIBRARIES}
)
