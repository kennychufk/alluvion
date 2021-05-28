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

add_executable(developing_hagen_poiseuille
  developing_hagen_poiseuille.cu
)

target_include_directories(developing_hagen_poiseuille
  PRIVATE
    ${FREETYPE_INCLUDE_DIRS}
)

target_link_libraries(developing_hagen_poiseuille
  PRIVATE
    lib_alluvion
    ${FREETYPE_LIBRARIES}
)

