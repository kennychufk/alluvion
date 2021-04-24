add_executable(test_alluvion
  main.cpp
  store_op.cu
  runner_test.cu
)
target_compile_features(test_alluvion PRIVATE cxx_std_11)
target_link_libraries(test_alluvion
  PRIVATE
    doctest::doctest
    lib_alluvion
)

set_target_properties(test_alluvion
  PROPERTIES
    CUDA_ARCHITECTURES OFF
)
