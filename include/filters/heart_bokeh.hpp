#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Launches the CUDA heart-shaped bokeh filter.
//
// TODO: Keep the CPU and CUDA versions' threshold, intensity, border, and
// rounding behavior identical so their results can be compared directly.
void launchHeartBokeh(
    ConstImageView input,
    ImageView output,
    unsigned char brightnessThreshold,
    float intensity,
    cudaStream_t stream);
