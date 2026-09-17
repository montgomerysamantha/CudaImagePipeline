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
    src/filters/heart_bokeh.cu
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

add_library(image_references
    src/reference/grayscale_cpu.cpp
    src/reference/gaussian_blur_cpu.cpp
    src/reference/sobel_cpu.cpp
    src/reference/heart_bokeh_cpu.cpp
)

target_include_directories(image_references
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cuda_image_pipeline src/main.cpp)
target_link_libraries(cuda_image_pipeline PRIVATE image_pipeline)

add_executable(
    benchmark_gaussian_blur
    benchmarks/benchmark_gaussian_blur.cu
)

target_link_libraries(
    benchmark_gaussian_blur
    PRIVATE
        image_pipeline
        image_references
)

add_executable(
    benchmark_grayscale
    benchmarks/benchmark_grayscale.cu
)

target_link_libraries(
    benchmark_grayscale
    PRIVATE
        image_pipeline
        image_references
)

add_executable(
    benchmark_sobel
    benchmarks/benchmark_sobel.cu
)

target_link_libraries(
    benchmark_sobel
    PRIVATE
        image_pipeline
        image_references
)

set_target_properties(
    benchmark_gaussian_blur
    benchmark_grayscale
    benchmark_sobel
    PROPERTIES CUDA_RUNTIME_LIBRARY Shared
)

include(CTest)

add_executable(benchmark_pipeline benchmarks/benchmark_pipeline.cu)
target_link_libraries(benchmark_pipeline PRIVATE image_pipeline)
set_target_properties(benchmark_pipeline PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_executable(benchmark_heart_bokeh benchmarks/benchmark_heart_bokeh.cu)
target_link_libraries(benchmark_heart_bokeh PRIVATE image_pipeline image_references)
set_target_properties(benchmark_heart_bokeh PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

if(BUILD_TESTING)
    add_executable(test_host_image tests/test_host_image.cpp)
    target_include_directories(test_host_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME host_image COMMAND test_host_image)
    add_executable(test_heart_bokeh_cpu tests/test_heart_bokeh_cpu.cpp)
    target_link_libraries(test_heart_bokeh_cpu PRIVATE image_references)
    add_test(NAME heart_bokeh_cpu COMMAND test_heart_bokeh_cpu)
    add_executable(test_grayscale tests/test_grayscale.cu)
    add_executable(test_gaussian_blur tests/test_gaussian_blur.cu)
    add_executable(test_edge_detection tests/test_edge_detection.cu)
    add_executable(test_pipeline tests/test_pipeline.cu)
    add_executable(test_heart_bokeh tests/test_heart_bokeh.cu)
    add_executable(test_sharpen tests/test_sharpen.cu)
    add_executable(test_resize tests/test_resize.cu)

    set(cuda_test_targets
        test_grayscale
        test_gaussian_blur
        test_edge_detection
        test_pipeline
        test_heart_bokeh
        test_sharpen
        test_resize
    )

    foreach(test_target IN LISTS cuda_test_targets)
        target_link_libraries(
            ${test_target}
            PRIVATE
                image_pipeline
                image_references
        )
        set_target_properties(
            ${test_target}
            PROPERTIES CUDA_RUNTIME_LIBRARY Shared
        )
    endforeach()

    add_test(NAME grayscale COMMAND test_grayscale)
    add_test(NAME gaussian_blur COMMAND test_gaussian_blur)
    add_test(NAME edge_detection COMMAND test_edge_detection)
    add_test(NAME pipeline COMMAND test_pipeline)
    add_test(NAME heart_bokeh COMMAND test_heart_bokeh)
    add_test(NAME sharpen COMMAND test_sharpen)
    add_test(NAME resize COMMAND test_resize)
endif()
