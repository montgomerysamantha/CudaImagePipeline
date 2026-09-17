#include "tests/test_runner.hpp"
#include "filters/sharpen.hpp"
#include "reference/sharpen_cpu.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

HostImage runSharpen(const HostImage& input, float strength = 1.0f)
{
    return test::runFilter(
        input,
        [strength](ConstImageView in, ImageView out, cudaStream_t stream)
        {
            launchSharpen(in, out, strength, stream);
        });
}

HostImage runSharpenCpu(const HostImage& input, float strength = 1.0f)
{
    HostImage output;
    output.allocate(input.width, input.height, input.channels);

    reference::launchSharpenCPU(
        input.view(),
        output.view(),
        strength
    );

    return output;
}

void testOnePixelImage()
{
    // Verify that the sharpening at strength 1.0 leaves a 1 x 1 image
    // unchanged.
    // Arrange
    constexpr int width = 1;
    constexpr int height = 1;

    const HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{187, 99, 190}
    );

    const HostImage expected = input;

    // Act
    const HostImage actual = runSharpen(input);
    const HostImage cpuOutput = runSharpenCpu(input);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testOnePixelImage"
    );

    test::requirePixelsEqual(
        cpuOutput,
        expected,
        "CPU sharpen: One pixel"
    );
}

void testSolidColorPreserved()
{
    // Verify that the sharpening at strength 1.0
    // leaves a solid color 5 x 5 image unchanged.
    // Arrange
    constexpr int width = 5;
    constexpr int height = 5;

    const HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{252, 186, 3}
    );

    const HostImage expected = input;

    // Act
    const HostImage actual = runSharpen(input);
    const HostImage cpuOutput = runSharpenCpu(input);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testSolidColorPreserved"
    );

    test::requirePixelsEqual(
        cpuOutput,
        expected,
        "CPU sharpen: Solid color"
    );
}

void testSharpenOutputValues()
{
    // Verify that the sharpening of strength 1.0 changes an image from this:
    //          0  1   2   3  4
    //      0  90 90  90  90 90
    //      1  90 90 100  90 90
    //      2  90 90  90  90 90
    //
    // To this expected result image:
    //          0  1   2   3  4
    //      0  90 90  80  90 90
    //      1  90 80 140  80 90
    //      2  90 90  80  90 90
    // Arrange
    constexpr int width = 5;
    constexpr int height = 3;

    HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{90, 90, 90}
    );

    test::setGrayscalePixel(input, 2, 1, 100);

    HostImage expected = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{90, 90, 90}
    );

    test::setGrayscalePixel(expected, 2, 1, 140); // Center
    test::setGrayscalePixel(expected, 2, 0, 80);  // North
    test::setGrayscalePixel(expected, 2, 2, 80);  // South
    test::setGrayscalePixel(expected, 1, 1, 80);  // West
    test::setGrayscalePixel(expected, 3, 1, 80);  // East

    // Act
    const HostImage actual = runSharpen(input);
    const HostImage cpuOutput = runSharpenCpu(input);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testSharpenOutputValues"
    );

    test::requirePixelsEqual(
        cpuOutput,
        expected,
        "CPU sharpen: Known impulse"
    );
}

void testBrightCenterClampsTo255()
{
    // Verify that the sharpening of strength 1.0 changes an image from this:
    //          0  1   2
    //      0  0   0   0
    //      1  0  255  0
    //      2  0   0   0
    //
    // At strength 1.0, the center is calculated to be 255 + (4 × 255 − 0 − 0 − 0 − 0) = 1275.
    // However, this calculation is expected to be clamped to 255.
    // Also, each of the four neighbors calculates 0 + (4 × 0 − 255) = −255, which is
    // clamped to 0 in the final image.
    // Expected result image:
    //          0  1   2
    //      0  0   0   0
    //      1  0  255  0
    //      2  0   0   0
    // Arrange
    constexpr int width = 3;
    constexpr int height = 3;

    HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{0, 0, 0}
    );

    test::setGrayscalePixel(input, 1, 1, 255);

    HostImage expected = input;
    // Act
    const HostImage actual = runSharpen(input);
    const HostImage cpuOutput = runSharpenCpu(input);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testBrightCenterClampsTo255"
    );

    test::requirePixelsEqual(
        cpuOutput,
        expected,
        "CPU sharpen: Bright clamp"
    );
}

void testDarkCenterClampsTo0()
{
    // Verify that the sharpening of strength 1.0 changes an image from this:
    //          0   1   2
    //      0  255 255 255
    //      1  255  0  255
    //      2  255 255 255
    //
    // At strength 1.0, the center is calculated to be 0 + (4 * 0 - 255 - 255 - 255 - 255) = -1020.
    // However, this calculation is expected to be clamped to 0.
    // Also, each of the four neighbors calculates 255 + (4 * 255 - 0 - 255 - 255 - 255) = 510, which is
    // clamped to 255 in the final image.
    // Expected result image:
    //          0   1   2
    //      0  255 255 255
    //      1  255  0  255
    //      2  255 255 255
    // Arrange
    constexpr int width = 3;
    constexpr int height = 3;

    HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{255, 255, 255}
    );

    test::setGrayscalePixel(input, 1, 1, 0);

    HostImage expected = input;
    // Act
    const HostImage actual = runSharpen(input);
    const HostImage cpuOutput = runSharpenCpu(input);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testDarkCenterClampsTo0"
    );

    test::requirePixelsEqual(
        cpuOutput,
        expected,
        "CPU sharpen: Dark clamp"
    );
}

void testZeroStrengthSharpen()
{
    const HostImage input = test::makeCoordinatePatternImage(17, 19);

    const HostImage actual = runSharpen(input, 0.0f);
    const HostImage cpuOutput = runSharpenCpu(input, 0.0f);

    test::requirePixelsEqual(
        actual, input, "testZeroStrengthSharpen");

    test::requirePixelsEqual(
        cpuOutput,
        actual,
        "CPU sharpen: Zero strength"
    );
}

void testSharpenMatchesCpuReference()
{
    // Exercise partial CUDA blocks and borders with varying RGB values.
    // Arrange
    const HostImage input = test::makeCoordinatePatternImage(17, 19);

    for (float strength : {0.0f, 0.5f, 1.0f})
    {
        // Act
        const HostImage expected = runSharpenCpu(input, strength);
        const HostImage actual = runSharpen(input, strength);

        // Assert
        test::requirePixelsEqual(
            actual,
            expected,
            "CPU/GPU sharpen agreement at strength "
                + std::to_string(strength)
        );
    }
}

int main()
{
    try
    {
        test::Runner runner;

        runner.run("One pixel", testOnePixelImage);
        runner.run("Solid color", testSolidColorPreserved);
        runner.run("Known output", testSharpenOutputValues);
        runner.run("Bright center clamp", testBrightCenterClampsTo255);
        runner.run("Dark center clamp", testDarkCenterClampsTo0);
        runner.run("Zero strength", testZeroStrengthSharpen);
        runner.run("CPU reference agreement", testSharpenMatchesCpuReference);

        runner.summary("Sharpen");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Sharpen test failed: " << error.what() << '\n';
        return 1;
    }
}