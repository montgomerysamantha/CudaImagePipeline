#include "tests/test_runner.hpp"
#include "filters/gaussian_blur.hpp"
#include "reference/gaussian_blur_cpu.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

void testSolidColorRemainsUnchanged()
{
    constexpr int width = 19;
    constexpr int height = 17;

    const HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{25, 100, 225}
    );

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
    // Arrange
    constexpr int width = 5;
    constexpr int height = 5;

    HostImage input = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{0, 0, 0}
    );

    test::setGrayscalePixel(input, 2, 2, 255);

    HostImage expected = test::makeSolidRgbImage(
        width,
        height,
        test::RgbPixel{0, 0, 0}
    );

    // Set expected pixels
    test::setGrayscalePixel(expected, 1, 1, 15);
    test::setGrayscalePixel(expected, 2, 1, 31);
    test::setGrayscalePixel(expected, 3, 1, 15);

    test::setGrayscalePixel(expected, 1, 2, 31);
    test::setGrayscalePixel(expected, 2, 2, 63);
    test::setGrayscalePixel(expected, 3, 2, 31);

    test::setGrayscalePixel(expected, 1, 3, 15);
    test::setGrayscalePixel(expected, 2, 3, 31);
    test::setGrayscalePixel(expected, 3, 3, 15);

    // Act
    const HostImage actual =
        test::runFilter(input, launchGaussianBlur);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testKnownImpulsePattern"
    );
}

void testAgainstCpuReference()
{
    // Arrange
    constexpr int width = 19;
    constexpr int height = 17;

    const HostImage input =
        test::makeCoordinatePatternImage(width, height);

    HostImage expected = input;

    reference::gaussianBlur(
        input.view(),
        expected.view()
    );

    // Act
    const HostImage actual =
        test::runFilter(
            input,
            launchGaussianBlur
        );

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testAgainstCpuReference"
    );
}

void testOnePixelImage()
{
    // Verify that the border weight normalization leaves a 1 x 1 image
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
    const HostImage actual =
        test::runFilter(
            input,
            launchGaussianBlur
        );

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testOnePixelImage"
    );
}

void testInvalidImageShapeIsRejected()
{
    // Arrange
    test::CudaStream stream;

    DeviceImage deviceInput(3, 3, 3);
    DeviceImage deviceOutput(4, 3, 3);

    const DeviceImage& readOnlyInput = deviceInput;

    bool exceptionWasThrown = false;

    // Act
    try
    {
        // Call launchGaussianBlur with the mismatched image views
        launchGaussianBlur(readOnlyInput.view(), deviceOutput.view(), stream.get());
    }
    catch (const std::invalid_argument&)
    {
        exceptionWasThrown = true;
    }

    // Assert
    test::require(
        exceptionWasThrown,
        "testInvalidImageShapeIsRejected: expected std::invalid_argument"
    );
}

int main()
{
    try
    {
        test::Runner runner;
        runner.run("testSolidColorRemainsUnchanged", testSolidColorRemainsUnchanged);
        runner.run("testKnownImpulsePattern", testKnownImpulsePattern);
        runner.run("testAgainstCpuReference", testAgainstCpuReference);
        runner.run("testOnePixelImage", testOnePixelImage);
        runner.run("testInvalidImageShapeIsRejected", testInvalidImageShapeIsRejected);
        runner.summary("Gaussian blur");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Gaussian blur test failed: " << error.what() << '\n';
        return 1;
    }
}
