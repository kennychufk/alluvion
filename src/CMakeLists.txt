pybind11_add_module(_alluvion
  pyalluvion.cu
)

target_link_libraries(_alluvion
  PUBLIC
    lib_alluvion
    Eigen3::Eigen
)

add_executable(display_test
  display_test.cu
)

target_link_libraries(display_test
  PRIVATE
    lib_alluvion
)

