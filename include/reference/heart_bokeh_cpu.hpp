#pragma once

#include "core/image.hpp"

namespace reference
{

// CPU reference for the heart-shaped bokeh filter.
//
// TODO: Implement this first, then use its output as the expected result for
// the CUDA implementation.
void heartBokeh(
    ConstImageView input,
    ImageView output,
    unsigned char brightnessThreshold,
    float intensity);

} // namespace reference
