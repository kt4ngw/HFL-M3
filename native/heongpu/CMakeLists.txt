# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(HFLMPrivateGramGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HEonGPU 1.1 REQUIRED)

set(NTL_ROOT "" CACHE PATH "Prefix containing the NTL headers and library")

add_executable(private_gram_gpu private_gram_gpu.cpp)
target_link_libraries(
    private_gram_gpu
    PRIVATE
        HEonGPU::heongpu
        CUDA::cudart
        OpenMP::OpenMP_CXX
)
set_target_properties(
    private_gram_gpu
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE OFF
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
if(NTL_ROOT)
    target_include_directories(private_gram_gpu PRIVATE "${NTL_ROOT}/include")
    target_link_directories(private_gram_gpu PRIVATE "${NTL_ROOT}/lib")
    target_link_options(
        private_gram_gpu
        PRIVATE "-Wl,-rpath,${NTL_ROOT}/lib"
    )
endif()
target_compile_options(
    private_gram_gpu
    PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
)
