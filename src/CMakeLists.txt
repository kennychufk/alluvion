add_executable(display_test
  display_test.cu
)

target_link_libraries(display_test
  PRIVATE
    lib_alluvion
)

add_executable(stationary
  stationary.cu
)

target_link_libraries(stationary
  PRIVATE
    lib_alluvion
)

