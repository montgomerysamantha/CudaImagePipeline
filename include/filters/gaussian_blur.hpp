#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

void launchGaussianBlur(ConstImageView input, ImageView output, cudaStream_t stream);

