#include "filters/edge_detection.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void testSolidGrayscaleImageHasNoEdges()
{
    constexpr int width = 19;
    constexpr int height = 17;

    const std::vector<unsigned char> grayPixels(
        static_cast<std::size_t>(width) * height * 3,
        120
    );

    const HostImage input =
        test::makeRgbImage(width, height, grayPixels);

    const HostImage expected = test::makeRgbImage(
        width,
        height,
        std::vector<unsigned char>(grayPixels.size(), 0)
    );

    const HostImage globalOutput =
        test::runFilter(input, launchEdgeDetection);

    const HostImage sharedOutput =
        test::runFilter(input, launchEdgeDetectionSharedMemory);

    test::requirePixelsEqual(
        globalOutput,
        expected,
        "global testSolidGrayscaleImageHasNoEdges"
    );

    test::requirePixelsEqual(
        sharedOutput,
        expected,
        "shared testSolidGrayscaleImageHasNoEdges"
    );
}

void testKnownVerticalEdge()
{
    // Arrange
    const HostImage input = test::makeGeneratedRgbImage(
        5,
        5,
        [](int x, int)
        {
            const unsigned char value = x < 2 ? 0 : 255;
            return test::RgbPixel{value, value, value};
        }
    );

    HostImage expected =
        test::makeSolidRgbImage(5, 5, {0, 0, 0});

    for (int y = 1; y < 4; ++y)
    {
        test::setGrayscalePixel(expected, 1, y, 255);
        test::setGrayscalePixel(expected, 2, y, 255);
    }

    // Act
    const HostImage globalOutput =
        test::runFilter(input, launchEdgeDetection);

    const HostImage sharedOutput =
        test::runFilter(input, launchEdgeDetectionSharedMemory);

    // Assert
    test::requirePixelsEqual(
        globalOutput,
        expected,
        "global testKnownVerticalEdge"
    );

    test::requirePixelsEqual(
        sharedOutput,
        expected,
        "shared testKnownVerticalEdge"
    );
}

void testBorderPixelsAreBlack()
{
    // Arrange
    const HostImage input = test::makeGeneratedRgbImage(
        7,
        6,
        [](int x, int y)
        {
            const unsigned char value =
                static_cast<unsigned char>((x * 31 + y * 47) % 256);

            return test::RgbPixel{value, value, value};
        }
    );

    // Act
    const HostImage globalOutput =
        test::runFilter(input, launchEdgeDetection);

    const HostImage sharedOutput =
        test::runFilter(input, launchEdgeDetectionSharedMemory);

    // Assert
    const auto requireBlackBorder =
        [](const HostImage& image, const std::string& testName)
        {
            for (int y = 0; y < image.height; ++y)
            {
                for (int x = 0; x < image.width; ++x)
                {
                    const bool isBorder =
                        x == 0 || y == 0 ||
                        x == image.width - 1 ||
                        y == image.height - 1;

                    if (!isBorder)
                    {
                        continue;
                    }

                    const int index = (y * image.width + x) * 3;
                    test::require(
                        image.pixels[index] == 0 &&
                        image.pixels[index + 1] == 0 &&
                        image.pixels[index + 2] == 0,
                        testName + ": a border pixel was not black"
                    );
                }
            }
        };

    requireBlackBorder(globalOutput, "global testBorderPixelsAreBlack");
    requireBlackBorder(sharedOutput, "shared testBorderPixelsAreBlack");
}

void testGlobalAndSharedImplementationsMatch()
{
    // Arrange
    const HostImage input = test::makeGeneratedRgbImage(
        17,
        19,
        [](int x, int y)
        {
            const unsigned char value =
                static_cast<unsigned char>((x * 13 + y * 7) % 256);

            return test::RgbPixel{value, value, value};
        }
    );

    // Act
    const HostImage globalOutput =
        test::runFilter(input, launchEdgeDetection);

    const HostImage sharedOutput =
        test::runFilter(input, launchEdgeDetectionSharedMemory);

    // Assert
    test::requirePixelsEqual(
        sharedOutput,
        globalOutput,
        "testGlobalAndSharedImplementationsMatch"
    );
}

void testInvalidImageShapeIsRejected()
{
    // Arrange
    test::CudaStream stream;
    DeviceImage deviceInput(3, 3, 3);
    DeviceImage deviceOutput(4, 3, 3);
    const DeviceImage& readOnlyInput = deviceInput;

    bool globalExceptionWasThrown = false;
    bool sharedExceptionWasThrown = false;

    // Act
    try
    {
        launchEdgeDetection(
            readOnlyInput.view(),
            deviceOutput.view(),
            stream.get()
        );
    }
    catch (const std::invalid_argument&)
    {
        globalExceptionWasThrown = true;
    }

    try
    {
        launchEdgeDetectionSharedMemory(
            readOnlyInput.view(),
            deviceOutput.view(),
            stream.get()
        );
    }
    catch (const std::invalid_argument&)
    {
        sharedExceptionWasThrown = true;
    }

    // Assert
    test::require(
        globalExceptionWasThrown,
        "testInvalidImageShapeIsRejected: global launcher did not reject the shape"
    );

    test::require(
        sharedExceptionWasThrown,
        "testInvalidImageShapeIsRejected: shared launcher did not reject the shape"
    );
}

int main()
{
    try
    {
        testSolidGrayscaleImageHasNoEdges();
        testKnownVerticalEdge();
        testBorderPixelsAreBlack();
        testGlobalAndSharedImplementationsMatch();
        testInvalidImageShapeIsRejected();
        std::cout << "Edge-detection tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Edge-detection test failed: " << error.what() << '\n';
        return 1;
    }
}
