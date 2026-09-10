#pragma once

#include "core/image.hpp"

namespace reference
{

// Adds bright neighbors through a fixed 21x20 heart aperture (anchor column 10,
// row 8). Threshold is inclusive, using integer RGB luminance in [0,255].
// Intensity must be finite and in [0,1]; zero preserves the input.
// This is additive, unnormalized light: overlapping highlights may saturate.
// Each scaled sum is rounded to nearest, added to the original, and capped at 255.
// Views must describe positive, equally sized, tightly packed RGB images with
// non-overlapping storage. Caller owns sufficiently large buffers.
// Throws std::invalid_argument for invalid views, overlap, or intensity.
void heartBokeh(
    ConstImageView input,
    ImageView output,
    unsigned char brightnessThreshold,
    float intensity);

} // namespace reference
