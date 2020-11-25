cmake_minimum_required(VERSION 3.0)
project(Rotate CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

add_executable(Rotate main.cu Pixel.cu Image.cu)

set_target_properties(
        Rotate
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Rotate cudart)