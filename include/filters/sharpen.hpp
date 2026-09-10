#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

// Template extension point: strength can scale an unsharp-mask or convolution kernel.
void launchSharpen(ConstImageView input, ImageView output, float strength, cudaStream_t stream);

