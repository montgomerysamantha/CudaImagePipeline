#include "filters/sharpen.hpp"
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

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testOnePixelImage"
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

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testSolidColorPreserved"
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

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testSharpenOutputValues"
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

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testBrightCenterClampsTo255"
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

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testDarkCenterClampsTo0"
    );
}

void testZeroStrengthSharpen()
{
    const HostImage input = test::makeCoordinatePatternImage(17, 19);

    const HostImage actual = runSharpen(input, 0.0f);

    test::requirePixelsEqual(
        actual, input, "testZeroStrengthSharpen");
}

int main()
{
    try
    {
        int passed = 0;
        auto run = [&](const char* name, auto testFunction)
        {
            testFunction();
            ++passed;
            std::cout << "PASS: " << name << '\n';
        };

        run("One pixel", testOnePixelImage);
        run("Solid color", testSolidColorPreserved);
        run("Known output", testSharpenOutputValues);
        run("Bright center clamp", testBrightCenterClampsTo255);
        run("Dark center clamp", testDarkCenterClampsTo0);
        run("Zero strength", testZeroStrengthSharpen);

        std::cout << "Sharpen: " << passed << " tests passed!\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Sharpen test failed: " << error.what() << '\n';
        return 1;
    }
}