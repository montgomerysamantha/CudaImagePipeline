#include "filters/grayscale.hpp"
#include "reference/grayscale_cpu.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

void testKnownRgbPixels()
{
    const HostImage input = test::makeRgbImage(
        2,
        2,
        {
            255,   0,   0,
              0, 255,   0,
              0,   0, 255,
            255, 255, 255
        }
    );

    const HostImage expected = test::makeRgbImage(
        2,
        2,
        {
             76,  76,  76,
            149, 149, 149,
             29,  29,  29,
            255, 255, 255
        }
    );

    const HostImage actual =
        test::runFilter(input, launchGrayscale);

    test::requirePixelsEqual(actual, expected, "testKnownRgbPixels");
}

void testAlreadyGrayscalePixelsRemainUnchanged()
{
    // Arrange
    const HostImage input = test::makeRgbImage(
        3,
        2,
        {
              0,   0,   0,
             42,  42,  42,
            127, 127, 127,
            180, 180, 180,
            240, 240, 240,
            255, 255, 255
        }
    );

    // Act
    const HostImage actual =
        test::runFilter(input, launchGrayscale);

    // Assert
    test::requirePixelsEqual(
        actual,
        input,
        "testAlreadyGrayscalePixelsRemainUnchanged"
    );
}

void testDimensionsOutsideBlockSize()
{
    // Arrange
    const HostImage input =
        test::makeCoordinatePatternImage(17, 19);

    HostImage expected =
        test::makeSolidRgbImage(17, 19, {0, 0, 0});

    reference::grayscale(input.view(), expected.view());

    // Act
    const HostImage actual =
        test::runFilter(input, launchGrayscale);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testDimensionsOutsideBlockSize"
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
        launchGrayscale(
            readOnlyInput.view(),
            deviceOutput.view(),
            stream.get()
        );
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
        testKnownRgbPixels();
        testAlreadyGrayscalePixelsRemainUnchanged();
        testDimensionsOutsideBlockSize();
        testInvalidImageShapeIsRejected();
        std::cout << "Grayscale tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Grayscale test failed: " << error.what() << '\n';
        return 1;
    }
}
