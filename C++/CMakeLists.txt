cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(iris_libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "C:/libtorch")

find_package(Torch REQUIRED)

add_executable(iris_libtorch main.cpp)
target_link_libraries(iris_libtorch "${TORCH_LIBRARIES}")