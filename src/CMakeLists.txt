add_executable(stationary
  stationary.cu
)

target_link_libraries(stationary
  PRIVATE
    lib_alluvion
)

add_executable(hagen_poiseuille
  hagen_poiseuille.cu
)

target_link_libraries(hagen_poiseuille
  PRIVATE
    lib_alluvion
)

