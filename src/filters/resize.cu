#include "filters/resize.hpp"

#include <stdexcept>

/*
width = 5
height = 1
[A B C D E]

sourceX = floor(outputX × inputWidth  / outputWidth)
sourceY = floor(outputY × inputHeight / outputHeight)

Which source indices do output positions 0, 1, and 2 select?
sourceX = floor(outputX × 0 / outputWidth)
sourceY = floor(outputY × 0 / outputHeight)

A

sourceX = floor(outputX × 0 / outputWidth)
sourceY = floor(outputY × 1 / outputHeight)

B

sourceX = floor(outputX × 0 / outputWidth)
sourceY = floor(outputY × 2 / outputHeight)

C
*/
void launchResize(ConstImageView input, ImageView output, cudaStream_t stream)
{
    (void)input;
    (void)output;
    (void)stream;

    // TODO: Map each output pixel into input coordinates and implement nearest,
    // bilinear, or bicubic sampling. The pipeline must allocate output to its new size.
    throw std::logic_error("Resize is a template and has not been implemented yet");
}

