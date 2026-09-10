#include "filters/resize.hpp"

#include <stdexcept>

void launchResize(ConstImageView input, ImageView output, cudaStream_t stream)
{
    (void)input;
    (void)output;
    (void)stream;

    // TODO: Map each output pixel into input coordinates and implement nearest,
    // bilinear, or bicubic sampling. The pipeline must allocate output to its new size.
    throw std::logic_error("Resize is a template and has not been implemented yet");
}

