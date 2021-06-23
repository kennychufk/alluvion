add_executable(stationary
  stationary.cu
)

target_link_libraries(stationary
  PRIVATE
    lib_alluvion
)

add_executable(stationary_dfsph
  stationary_dfsph.cu
)

target_link_libraries(stationary_dfsph
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

add_executable(developing_hagen_poiseuille
  developing_hagen_poiseuille.cu
)

target_link_libraries(developing_hagen_poiseuille
  PRIVATE
    lib_alluvion
)

add_executable(stabilizer
  stabilizer.cu
)

target_link_libraries(stabilizer
  PRIVATE
    lib_alluvion
)

