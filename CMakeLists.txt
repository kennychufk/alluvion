cmake_minimum_required(VERSION 3.20.0)
project(alluvion
  LANGUAGES CXX C CUDA
)
set (CMAKE_CUDA_STANDARD 17)

find_package(doctest REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
option(ALLUVION_BUILD_EXECUTABLES "build C++ executables for alluvion" OFF)
option(ALLUVION_BUILD_TESTS "build tests for alluvion" OFF)

add_subdirectory(lib)
if (ALLUVION_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if (ALLUVION_BUILD_EXECUTABLES)
  add_subdirectory(src)
endif()
