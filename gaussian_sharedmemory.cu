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
const int RUNS = 20;

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

__global__ void grayscaleGpu(int width, int height, unsigned char* image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel = y * width + x;
        int pixelIndex = pixel * 3;

        // Read R, G, B
        int r = image[pixelIndex];
        int g = image[pixelIndex + 1];
        int b = image[pixelIndex + 2];

        // Calculate gray values
        int gray = calcGray(r, g, b);

        // Write gray back to R, G, B
        image[pixelIndex] = gray;
        image[pixelIndex + 1] = gray;
        image[pixelIndex + 2] = gray;
    }
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

__device__ bool checkBounds(int x, int y, int width, int height)
{
    if (x >= 0 && x < width &&
        y >= 0 && y < height)
    {
        return true;
    }
    return false;
}

__device__ void fillSharedTile(unsigned char tile[18][18][3], int sharedX, int sharedY, int globalX, int globalY, int width, const unsigned char* input)
{
    int pixel = globalY * width + globalX;
    int pixelIndex = pixel * 3;
    tile[sharedY][sharedX][0] = input[pixelIndex];
    tile[sharedY][sharedX][1] = input[pixelIndex + 1];
    tile[sharedY][sharedX][2] = input[pixelIndex + 2];
}

__device__ void loadHalos(
    unsigned char tile[18][18][3],
    int x,
    int y,
    int sharedX,
    int sharedY,
    int width,
    int height,
    const unsigned char* input)
{
    // left edge threads
    if (threadIdx.x == 0)
    {
        int haloX = x - 1;
        int haloY = y;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX - 1, sharedY, haloX, haloY, width, input);
        }
    }

    // right edge threads
    if (threadIdx.x == blockDim.x - 1)
    {
        int haloX = x + 1;
        int haloY = y;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX + 1, sharedY, haloX, haloY, width, input);
        }
    }

    // top left corner
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int haloX = x - 1;
        int haloY = y - 1;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX - 1, sharedY - 1, haloX, haloY, width, input);
        }
    }

    // top right corner
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
    {
        int haloX = x + 1;
        int haloY = y - 1;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX + 1, sharedY - 1, haloX, haloY, width, input);
        }
    }

    // top threads
    if (threadIdx.y == 0)
    {
        int haloX = x;
        int haloY = y - 1;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX, sharedY - 1, haloX, haloY, width, input);
        }
    }

    // bottom left corner
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
    {
        int haloX = x - 1;
        int haloY = y + 1;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX - 1, sharedY + 1, haloX, haloY, width, input);
        }
    }

    // bottom right corner
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
    {
        int haloX = x + 1;
        int haloY = y + 1;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX + 1, sharedY + 1, haloX, haloY, width, input);
        }
    }

    // bottom threads
    if (threadIdx.y == blockDim.y - 1)
    {
        int haloX = x;
        int haloY = y + 1;

        if (checkBounds(haloX, haloY, width, height))
        {
            fillSharedTile(tile, sharedX, sharedY + 1, haloX, haloY, width, input);
        }
    }
}

__global__ void gaussianBlurGpuShared(int width, int height, const unsigned char* input, unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    __shared__ unsigned char tile[18][18][3];

    if (checkBounds(x, y, width, height))
    {
        fillSharedTile(
            tile,
            sharedX,
            sharedY,
            x,
            y,
            width,
            input
        );
    }

    loadHalos(
        tile,
        x,
        y,
        sharedX,
        sharedY,
        width,
        height,
        input);

    __syncthreads();

    if (x < width && y < height)
    {
        int rSum = 0;
        int gSum = 0;
        int bSum = 0;
        int weightSum = 0;

        int kernel[3][3] =
        {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int neighborX = x + dx;
                int neighborY = y + dy;
                if (checkBounds(neighborX, neighborY, width, height))
                {
                    int weight = kernel[dy + 1][dx + 1];
                    int r = tile[sharedY + dy][sharedX + dx][0];
                    int g = tile[sharedY + dy][sharedX + dx][1];
                    int b = tile[sharedY + dy][sharedX + dx][2];

                    rSum += r * weight;
                    gSum += g * weight;
                    bSum += b * weight;

                    weightSum += weight;
                }
            }
        }

        int pixel = y * width + x;
        int pixelIndex = pixel * 3;

        output[pixelIndex] = rSum / weightSum;
        output[pixelIndex + 1] = gSum / weightSum;
        output[pixelIndex + 2] = bSum / weightSum;
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

std::pair<float, float> gpuGaussianBenchmark(dim3 blocks, dim3 threadsPerBlock, int width, int height, unsigned char* d_input_image, unsigned char* d_output_image)
{
    float naiveTotal = 0.0f;
    float sharedTotal = 0.0f;

    // warmup
    gaussianBlurGpu<<<blocks, threadsPerBlock>>>(
        width, height, d_input_image, d_output_image
    );

    gaussianBlurGpuShared<<<blocks, threadsPerBlock>>>(
        width, height, d_input_image, d_output_image
    );

    cudaDeviceSynchronize();

    // benchmark naive
    for (int i = 0; i < RUNS; i++)
    {
        naiveTotal += timeCuda([&]()
        {
            gaussianBlurGpu<<<blocks, threadsPerBlock>>>(
                width,
                height,
                d_input_image,
                d_output_image
            );
        });
    }

    // benchmark shared
    for (int i = 0; i < RUNS; i++)
    {
        sharedTotal += timeCuda([&]()
        {
            gaussianBlurGpuShared<<<blocks, threadsPerBlock>>>(
                width,
                height,
                d_input_image,
                d_output_image
            );
        });
    }

    float naiveAvg = naiveTotal / RUNS;
    float sharedAvg = sharedTotal / RUNS;

    return {naiveAvg, sharedAvg};
}

float cpuGaussianBenchmark(int width, int height, const unsigned char* input, unsigned char* output)
{
    float cpuTotal = 0.0f;

    for (int i = 0; i < RUNS; i++)
    {
        cpuTotal += timeCpu([&]()
        {
            gaussianBlurCpu(width, height, input, output);
        });
    }

    float cpuAvg = cpuTotal / RUNS;
    return cpuAvg;
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

    /*
        load image

        allocate GPU memory
    */
    unsigned char* d_input_image;
    cudaMalloc((void**)&d_input_image, bytes);

    unsigned char* d_output_image;
    cudaMalloc((void**)&d_output_image, bytes);

    // copy rgb image from CPU → GPU, calc time taken
    // copy image → GPU
    float hostToDeviceTime = timeCuda([&]()
    {
    cudaMemcpy(d_input_image, gpu_image.data(), bytes, cudaMemcpyHostToDevice);
    });

    dim3 threadsPerBlock(16, 16);

    int blocksX = (width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    dim3 blocks(blocksX, blocksY);

    /*
        warm up GPU

        verify naive/shared correctness

        benchmark CPU × 20
        benchmark naive GPU × 20
        benchmark shared GPU × 20

        print comparison

        copy/save image if desired
        cleanup
    */


    // run gpu benchmarks
    auto [naiveAvg, sharedAvg] = gpuGaussianBenchmark(
        blocks,
        threadsPerBlock,
        width,
        height,
        d_input_image,
        d_output_image
    );

    // apply gaussian blur to image, calc time taken
    float gpuTime = timeCuda([&]()
    {
        gaussianBlurGpuShared<<<blocks, threadsPerBlock>>>(width, height, d_input_image, d_output_image);
    });

    // copy grayscale image from GPU → CPU, calc time taken
    float deviceToHostTime = timeCuda([&]()
    {
        cudaMemcpy(gpu_image.data(), d_output_image, bytes, cudaMemcpyDeviceToHost);
    });

    std::vector<unsigned char> cpu_image_blur(image, image + bytes);
    float cpuAvg = cpuGaussianBenchmark(width, height, cpu_image.data(), cpu_image_blur.data());

    verifyCpuGpuImages(cpu_image_blur.data(), gpu_image.data(), bytes);

    std::cout << "\nGaussian Blur — " << width << "x" << height << " RGB\n";

    std::cout << "CPU:                 " << cpuAvg << " ms\n";
    std::cout << "CUDA naive:          " << naiveAvg << " ms\n";
    std::cout << "CUDA shared memory:  " << sharedAvg << " ms\n";

    std::cout << "\nNaive -> shared:     "
            << naiveAvg / sharedAvg << "x\n";

    std::cout << "CPU -> shared:       "
            << cpuAvg / sharedAvg << "x\n";

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

    // free GPU memory and image loaded
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    stbi_image_free(image);

    return 0;
}