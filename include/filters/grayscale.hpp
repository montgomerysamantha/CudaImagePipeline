#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

void launchGrayscale(ConstImageView input, ImageView output, cudaStream_t stream);

