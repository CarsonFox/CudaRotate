cmake_minimum_required(VERSION 3.10)
project(Rotate CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

add_executable(Rotate main.cu Pixel.cu Image.cu)

set_target_properties(
        Rotate
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)