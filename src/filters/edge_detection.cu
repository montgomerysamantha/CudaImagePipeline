#include "filters/edge_detection.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>

namespace
{

__constant__ int sobelXDevice[3][3] =
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int sobelYDevice[3][3] =
{
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__device__ void writeRgbDevice(
    unsigned char value,
    unsigned char* output,
    int outputIndex)
{
    output[outputIndex] = value;
    output[outputIndex + 1] = value;
    output[outputIndex + 2] = value;
}

__global__ void sobelKernel(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

}

}
void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream)
{
    (void)input;
    (void)output;
    (void)stream;

    // TODO: Add a Sobel kernel here. Read only from input and write only to output,
    // then check the launch with CUDA_CHECK(cudaGetLastError()).
    throw std::logic_error("Edge detection is a template and has not been implemented yet");
}

