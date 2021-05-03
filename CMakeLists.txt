cmake_minimum_required(VERSION 3.1.0)
project(alluvion
  LANGUAGES CXX C CUDA
)

find_package(doctest REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(src)
