add_executable(test_alluvion
  main.cpp
  store_op.cu
  runner_test.cu
  field_op.cu
  field_gen.cu
)
target_link_libraries(test_alluvion
  PRIVATE
    doctest::doctest
    lib_alluvion
)

set_target_properties(test_alluvion
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
