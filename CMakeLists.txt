cmake_minimum_required(VERSION 3.22)

project(CudaLearning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(image_pipeline
    src/core/device_image.cu
    src/core/image_io.cpp
    src/filters/grayscale.cu
    src/filters/gaussian_blur.cu
    src/filters/edge_detection.cu
    src/filters/sharpen.cu
    src/filters/resize.cu
    src/pipeline/image_pipeline.cu
)

target_include_directories(image_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set_target_properties(image_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cuda_image_pipeline src/main.cpp)
target_link_libraries(cuda_image_pipeline PRIVATE image_pipeline)

add_executable(
    benchmark_gaussian_blur
    benchmarks/benchmark_gaussian_blur.cu
)

target_link_libraries(
    benchmark_gaussian_blur
    PRIVATE image_pipeline
)

add_executable(
    benchmark_grayscale
    benchmarks/benchmark_grayscale.cu
)

target_link_libraries(
    benchmark_grayscale
    PRIVATE image_pipeline
)

add_executable(
    benchmark_sobel
    benchmarks/benchmark_sobel.cu
)

target_link_libraries(
    benchmark_sobel
    PRIVATE image_pipeline
)

set_target_properties(
    benchmark_gaussian_blur
    benchmark_grayscale
    benchmark_sobel
    PROPERTIES CUDA_RUNTIME_LIBRARY Shared
)
