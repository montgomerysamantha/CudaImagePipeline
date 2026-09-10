#include "reference/heart_bokeh_cpu.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool value)
{
    if (!value)
    {
        throw std::runtime_error("Heart bokeh CPU check failed");
    }
}

HostImage makeBlackImage(int width, int height)
{
    HostImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(image.view().bytes(), 0);
    return image;
}

void testZeroIntensityPreservesInput()
{
    // Rectangular images catch swapped traversal dimensions.
    for (const auto dimensions : {std::pair<int, int>{47, 31}, {13, 41}, {1, 1}})
    {
        HostImage input = makeBlackImage(dimensions.first, dimensions.second);
        for (std::size_t byteIndex = 0; byteIndex < input.pixels.size(); byteIndex++)
            input.pixels[byteIndex] = static_cast<unsigned char>(byteIndex % 256);
        const HostImage &source = input;
        HostImage output = input;
        reference::heartBokeh(source.view(), output.view(), 0, 0);
        require(input.pixels == output.pixels);
    }
}

// Hand-picked tip, notch and brightness expectations are independent of the kernel.
void testImpulseAndValidation()
{
    HostImage input = makeBlackImage(47, 31);

    const HostImage &source = input;

    HostImage output = input;

    reference::heartBokeh(source.view(), output.view(), 200, 1);
    require(output.pixels == input.pixels);
    constexpr int centerRow = 15;
    constexpr int centerColumn = 23;
    constexpr int tipOffset = 11;
    constexpr int notchOffset = -8;
    constexpr int expectedOpenSamples = 282;

    const int center = (centerRow * input.width + centerColumn) * input.channels;

    for (int channel = 0; channel < 3; channel++)
        input.pixels[center + channel] = 255;

    reference::heartBokeh(source.view(), output.view(), 255, 1);

    int illuminatedPixelCount = 0;

    for (std::size_t byteIndex = 0; byteIndex < output.pixels.size(); byteIndex += 3)
        if (output.pixels[byteIndex] == 255)
            illuminatedPixelCount++;
    require(illuminatedPixelCount == expectedOpenSamples);
    require(output.pixels[((centerRow + tipOffset) * input.width + centerColumn) *
                          input.channels] == 255); // Bottom tip.
    require(output.pixels[((centerRow + notchOffset) * input.width + centerColumn) *
                          input.channels] == 0); // Top notch.
    reference::heartBokeh(source.view(), output.view(), 255, 0.5f);
    require(output.pixels[((centerRow + tipOffset) * input.width + centerColumn) *
                          input.channels] == 128);
    require(output.pixels[center] == 255); // Saturation.
    for (int channel = 0; channel < 3; channel++)
        input.pixels[center + channel] = 199;

    reference::heartBokeh(source.view(), output.view(), 200, 1);
    require(output.pixels == input.pixels);

    bool rejected = false;

    try
    {
        reference::heartBokeh(source.view(), input.view(), 200, 1);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected);
    rejected = false;

    try
    {
        reference::heartBokeh(source.view(), output.view(), 200,
                              std::numeric_limits<float>::quiet_NaN());
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected);
}

int main()
{
    try
    {
        testZeroIntensityPreservesInput();
        testImpulseAndValidation();
        std::cout << "Heart bokeh CPU checks passed\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
