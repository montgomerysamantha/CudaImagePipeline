#include "filters/sharpen.hpp"

#include <stdexcept>

void launchSharpen(ConstImageView input, ImageView output, float strength, cudaStream_t stream)
{
    (void)input;
    (void)output;
    (void)strength;
    (void)stream;

    // TODO: Add a 3x3 sharpen or unsharp-mask kernel. Keep input and output separate
    // so neighboring threads never consume values already modified by the kernel.
    throw std::logic_error("Sharpen is a template and has not been implemented yet");
}

