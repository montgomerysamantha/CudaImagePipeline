#include <iostream>
#include <chrono>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

const char* input_image = "lena.jpg";
const char* output_image = "lena_gaussian.png";

template <typename Func>
float timeCuda(Func func)
{
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    func();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedMs;
}

template <typename Func>
float timeCpu(Func func)
{
    auto cpuStart = std::chrono::high_resolution_clock::now();

    func();

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = cpuEnd - cpuStart;

    return static_cast<float>(duration.count());
}

__host__ __device__ int calcGray(int r, int g, int b)
{
    return (299 * r + 587 * g + 114 * b) / 1000;
}

__host__ __device__
void blurPixel(
    int x,
    int y,
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    int kernel[3][3] =
    {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    int pixel = y * width + x;
    int pixelIndex = pixel * 3;

    int rSum = 0;
    int gSum = 0;
    int bSum = 0;
    int weightSum = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int neighborX = x + dx;
            int neighborY = y + dy;

            if (neighborX >= 0 && neighborX < width &&
                neighborY >= 0 && neighborY < height)
            {
                int weight = kernel[dy + 1][dx + 1];

                int neighborPixel =
                    neighborY * width + neighborX;

                int neighborPixelIndex =
                    neighborPixel * 3;

                int r = input[neighborPixelIndex];
                int g = input[neighborPixelIndex + 1];
                int b = input[neighborPixelIndex + 2];

                rSum += r * weight;
                gSum += g * weight;
                bSum += b * weight;

                weightSum += weight;
            }
        }
    }

    output[pixelIndex] = rSum / weightSum;
    output[pixelIndex + 1] = gSum / weightSum;
    output[pixelIndex + 2] = bSum / weightSum;
}

__global__ void gaussianBlurGpu(int width, int height, const unsigned char* input, unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
       blurPixel(x, y, width, height, input, output);
    }
}

void gaussianBlurCpu(int width, int height, const unsigned char* input, unsigned char* output)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            blurPixel(x, y, width, height, input, output);
        }
    }
}


void grayscaleCpu(int width, int height, unsigned char* image)
{
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        // Find where pixel i starts in the RGB array
        int pixelIndex = i * 3;

        // Read R, G, B
        int r = image[pixelIndex];
        int g = image[pixelIndex + 1];
        int b = image[pixelIndex + 2];

        // Calculate gray
        int gray = calcGray(r, g, b);

        // Write gray back to R, G, B
        image[pixelIndex] = gray;
        image[pixelIndex + 1] = gray;
        image[pixelIndex + 2] = gray;
    }
}

bool verifyCpuGpuImages(unsigned char* cpu_image, unsigned char* gpu_image, size_t bytes)
{
    for (size_t i = 0; i < bytes; i++)
    {
        if (cpu_image[i] != gpu_image[i])
        {
            std::cout << "Mismatch at byte " << i
                      << ": CPU = " << static_cast<int>(cpu_image[i])
                      << ", GPU = " << static_cast<int>(gpu_image[i])
                      << std::endl;
            return false;
        }
    }

    std::cout << "CPU and GPU results match!" << std::endl;
    return true;
}

int main()
{
    int width;
    int height;
    int channels;

    unsigned char* image = stbi_load(
        input_image,
        &width,
        &height,
        &channels,
        3
    );

    if (!image)
    {
        std::cout << "Yo the pic failed to load" << std::endl;
        return -1;
    }

    int total_pixels = width * height;

    std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Total Pixels: " << total_pixels << std::endl;
    std::cout << "Original channels: " << channels << std::endl;

    size_t bytes = width * height * 3 * sizeof(unsigned char);
    std::vector<unsigned char> cpu_image(image, image + bytes);
    std::vector<unsigned char> gpu_image(image, image + bytes);

    unsigned char* d_input_image;
    cudaMalloc((void**)&d_input_image, bytes);

    unsigned char* d_output_image;
    cudaMalloc((void**)&d_output_image, bytes);

    // copy rgb image from CPU → GPU, calc time taken
    float hostToDeviceTime = timeCuda([&]()
    {
    cudaMemcpy(d_input_image, gpu_image.data(), bytes, cudaMemcpyHostToDevice);
    });

    dim3 threadsPerBlock(16, 16);

    int blocksX = (width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    dim3 blocks(blocksX, blocksY);

    // apply gaussian blur to image, calc time taken
    float gpuTime = timeCuda([&]()
    {
        gaussianBlurGpu<<<blocks, threadsPerBlock>>>(width, height, d_input_image, d_output_image);
    });

    // copy grayscale image from GPU → CPU, calc time taken
    float deviceToHostTime = timeCuda([&]()
    {
        cudaMemcpy(gpu_image.data(), d_output_image, bytes, cudaMemcpyDeviceToHost);
    });

    // save image
    int success = stbi_write_png(
        output_image,
        width,
        height,
        3,
        gpu_image.data(),
        width * 3
    );

    if (!success)
    {
        std::cout << "Failed to save gaussian image!" << std::endl;
    }
    else
    {
        std::cout << "Guassian image saved!" << std::endl;
    }

    std::vector<unsigned char> cpu_image_blur(image, image + bytes);
    float cpuTime = timeCpu([&]()
    {
        gaussianBlurCpu(width, height, cpu_image.data(), cpu_image_blur.data());
    });

    verifyCpuGpuImages(cpu_image_blur.data(), gpu_image.data(), bytes);

    // free GPU memory and image loaded
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    stbi_image_free(image);

    std::cout << "Host to Device Time:" << hostToDeviceTime << " ms\n";
    std::cout << "GPU Compute:        " << gpuTime << " ms\n";
    std::cout << "Device to Host Time:" << deviceToHostTime << " ms\n";
    std::cout << "GPU Total:          " << hostToDeviceTime + gpuTime + deviceToHostTime << " ms\n";
    std::cout << "CPU Total:        " << cpuTime << " ms\n";

    return 0;
}