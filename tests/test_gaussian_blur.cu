#include "filters/gaussian_blur.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <vector>

void testSolidColorRemainsUnchanged()
{
    constexpr int width = 19;
    constexpr int height = 17;

    std::vector<unsigned char> pixels(
        static_cast<std::size_t>(width) * height * 3
    );

    for (std::size_t index = 0; index < pixels.size(); index += 3)
    {
        pixels[index] = 25;
        pixels[index + 1] = 100;
        pixels[index + 2] = 225;
    }

    const HostImage input =
        test::makeRgbImage(width, height, pixels);

    const HostImage actual =
        test::runFilter(input, launchGaussianBlur);

    test::requirePixelsEqual(
        actual,
        input,
        "testSolidColorRemainsUnchanged"
    );
}

void testKnownImpulsePattern()
{
    // TODO: Put one white pixel in the center of a small black image. Calculate
    // the expected 3x3 weighted pattern by hand and compare every output pixel.
}

void testAgainstCpuReference()
{
    // TODO: Fill a 17 x 19 image with deterministic values, run a CPU blur
    // reference, and compare it with launchGaussianBlur.
}

void testOnePixelImage()
{
    // TODO: Verify that the border weight normalization leaves a 1 x 1 image
    // unchanged.
}

void testInvalidImageShapeIsRejected()
{
    // TODO: Verify mismatched input/output dimensions or channel counts throw.
}

int main()
{
    try
    {
        testSolidColorRemainsUnchanged();
        std::cout << "Gaussian blur tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Gaussian blur test failed: " << error.what() << '\n';
        return 1;
    }
}
