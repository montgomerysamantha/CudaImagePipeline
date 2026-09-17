#pragma once

#include "core/image.hpp"

namespace reference
{

void launchSharpenCPU(ConstImageView input, ImageView output, float strength);

} // namespace reference