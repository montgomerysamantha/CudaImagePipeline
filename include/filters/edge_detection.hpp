#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Input must contain grayscale pixels.
// Three-channel input is allowed when R, G, and B contain the same value.
// Uses the global-memory implementation selected for the production pipeline.
void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream);

// Experimental implementation retained for comparison and learning.
void launchEdgeDetectionSharedMemory(ConstImageView input, ImageView output, cudaStream_t stream);
