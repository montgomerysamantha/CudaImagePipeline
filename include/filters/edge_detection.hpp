#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Template extension point: implement a Sobel or Canny kernel in edge_detection.cu.
void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream);

