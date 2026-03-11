cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(iris_libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "C:/libtorch")

find_package(Torch REQUIRED)

add_executable(iris_libtorch main.cpp)
target_link_libraries(iris_libtorch "${TORCH_LIBRARIES}")

set_property(TARGET iris_libtorch PROPERTY CXX_STANDARD 17)

if(MSVC)
    file(GLOB TORCH_DLLS "${CMAKE_PREFIX_PATH}/lib/*.dll")
    add_custom_command(TARGET iris_libtorch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:iris_libtorch>)
endif()