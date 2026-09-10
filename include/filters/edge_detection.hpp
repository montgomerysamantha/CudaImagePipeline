#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Input must contain grayscale pixels.
// Three-channel input is allowed when R, G, and B contain the same value.
void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream);

void launchEdgeDetectionGlobalMemory(ConstImageView input, ImageView output, cudaStream_t stream);
