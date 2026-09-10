#include "core/image.hpp"

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <utility>

constexpr int sobelX[3][3] =
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

constexpr int sobelY[3][3] =
{
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

std::pair<int, int> getGxGyValues(
    int x,
    int y,
    int width,
    const unsigned char* input)
{
    int gx = 0;
    int gy = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;

            const int neighborIndex =
                (neighborY * width + neighborX) * 3;

            const int gray = input[neighborIndex];

            gx += gray * sobelX[dy + 1][dx + 1];
            gy += gray * sobelY[dy + 1][dx + 1];
        }
    }

    return {gx, gy};
}

unsigned char calcEdge(int gx, int gy)
{
    const float magnitude =
        std::sqrt(static_cast<float>(gx * gx + gy * gy));

    const unsigned char edge =
        static_cast<unsigned char>(
            std::min(255.0f, magnitude)
        );

    return edge;
}

bool isBorder(int x, int y, int width, int height)
{
    const bool isBorderCoord =
                x == 0 ||
                y == 0 ||
                x == width - 1 ||
                y == height - 1;

    return isBorderCoord;
}

void writeRGB(
    const unsigned char r,
    const unsigned char g,
    const unsigned char b,
    unsigned char* output,
    const int outputIndex)
{
    output[outputIndex] = r;
    output[outputIndex + 1] = g;
    output[outputIndex + 2] = b;
}

void sobelCpu(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    // Loop over every pixel.
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int outputIndex =
                (y * width + x) * 3;


            if (isBorder(x, y, width, height))
            {
                writeRGB(0, 0, 0, output, outputIndex);
                continue;
            }

            const auto [gx, gy] =
                getGxGyValues(
                    x,
                    y,
                    width,
                    input
                );

            const unsigned char edge =
                calcEdge(gx, gy);

            writeRGB(edge, edge, edge, output, outputIndex);
        }
    }
}

int main()
{
    try
    {
        // This image is already grayscale, which is what sobelCpu expects.
        const HostImage input =
            loadRgbImage("lena_grayscale.png");

        HostImage output;
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.pixels.resize(input.pixels.size());

        sobelCpu(
            input.width,
            input.height,
            input.pixels.data(),
            output.pixels.data()
        );

        std::filesystem::create_directories("output");

        savePngImage(
            "output/lena_sobel_cpu.png",
            output
        );

        std::cout
            << "Saved CPU Sobel result to "
            << "output/lena_sobel_cpu.png\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Sobel failed: "
            << error.what()
            << '\n';

        return 1;
    }
}