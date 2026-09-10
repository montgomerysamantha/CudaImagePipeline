#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream);

void launchEdgeDetectionGlobalMemory(ConstImageView input, ImageView output, cudaStream_t stream);

