#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Explicit variants for benchmarking and correctness comparisons.
void launchHeartBokehGlobal(ConstImageView, ImageView, unsigned char, float, cudaStream_t);
void launchHeartBokehShared(ConstImageView, ImageView, unsigned char, float, cudaStream_t);

// Launches the CUDA heart-shaped bokeh filter.
//
// Same additive 21x20 aperture, inclusive threshold, rounding and clamping as
// reference::heartBokeh. Requires positive equal RGB shapes, non-overlapping
// device buffers, and finite intensity in [0,1]. Invalid arguments throw.
// Launches asynchronously on stream; caller must synchronize before host reads.
void launchHeartBokeh(
    ConstImageView input,
    ImageView output,
    unsigned char brightnessThreshold,
    float intensity,
    cudaStream_t stream);
