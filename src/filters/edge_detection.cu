#include "filters/edge_detection.hpp"

#include <stdexcept>

void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream)
{
    (void)input;
    (void)output;
    (void)stream;

    // TODO: Add a Sobel kernel here. Read only from input and write only to output,
    // then check the launch with CUDA_CHECK(cudaGetLastError()).
    throw std::logic_error("Edge detection is a template and has not been implemented yet");
}

