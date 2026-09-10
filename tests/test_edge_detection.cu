#include "filters/edge_detection.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
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
    // TODO: Make the left half of a small image black and the right half white.
    // Calculate the expected Sobel response near the dividing line.
}

void testBorderPixelsAreBlack()
{
    // TODO: Use a non-uniform grayscale image and verify the first/last rows and
    // columns are black for both Sobel implementations.
}

void testGlobalAndSharedImplementationsMatch()
{
    // TODO: Use deterministic grayscale values in a 17 x 19 image and compare
    // every byte produced by the global- and shared-memory launchers.
}

void testInvalidImageShapeIsRejected()
{
    // TODO: Verify mismatched input/output dimensions or channels throw.
}

int main()
{
    try
    {
        testSolidGrayscaleImageHasNoEdges();
        std::cout << "Edge-detection tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Edge-detection test failed: " << error.what() << '\n';
        return 1;
    }
}
