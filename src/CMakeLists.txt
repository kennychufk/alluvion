add_executable(stationary
  stationary.cu
)

target_link_libraries(stationary
  PRIVATE
    lib_alluvion
)

