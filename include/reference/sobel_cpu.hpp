#pragma once

#include "core/image.hpp"

namespace reference
{

// Input must contain grayscale pixels stored as equal R, G, and B values.
void sobel(ConstImageView input, ImageView output);

} // namespace reference
