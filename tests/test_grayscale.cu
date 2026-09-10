#include "filters/grayscale.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>

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
    // TODO: Create an RGB-stored image whose R, G, and B values already match.
    // Run launchGrayscale and verify that every value remains unchanged.
}

void testDimensionsOutsideBlockSize()
{
    // TODO: Test an image such as 17 x 19. This exercises partial CUDA blocks.
}

void testInvalidImageShapeIsRejected()
{
    // TODO: Give the launcher mismatched input/output dimensions and verify that
    // it throws std::invalid_argument.
}

int main()
{
    try
    {
        testKnownRgbPixels();
        std::cout << "Grayscale tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Grayscale test failed: " << error.what() << '\n';
        return 1;
    }
}
