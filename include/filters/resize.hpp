#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Template extension point: output owns the requested destination dimensions.
void launchResize(ConstImageView input, ImageView output, cudaStream_t stream);

