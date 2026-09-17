#include "tests/test_runner.hpp"
#include "filters/resize.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

void testResizeBigger()
{
    // Verify that resize enlarges the pixels of an image:
    // A = (255, 0, 0)
    // B = (0, 255, 0)
    // C = (0, 0, 255)

    // Input:    [A B C]
    // Expected: [A A B B C]

    // Arrange
    const HostImage input = test::makeRgbImage(
        3, 1,
        {
            255,   0,    0, // A
            0,    255,   0,  // B
            0,     0,   255    // C
        }
    );

    constexpr int outputWidth = 5;
    constexpr int outputHeight = 1;

    const HostImage expected = test::makeRgbImage(
        outputWidth, outputHeight,
        {
            255,   0,    0,  // A
            255,   0,    0,  // A
            0,    255,   0,  // B
            0,    255,   0,  // B
            0,     0,   255  // C
        }
    );

    // Act
    const HostImage actual = test::runResize(input, outputWidth, outputHeight, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeBigger"
    );
}


// Learning outlines: implement each body, then uncomment its runner entry.
// Unfinished outlines are not registered, so they cannot count as passing tests.

void testResizeTaller()
{
    // Enlarge a 1 x 3 column [A B C] into a 1 x 5 column [A A B B C].
    // Verify that resize enlarges the pixels of an image:
    // A = (255, 0, 0)
    // B = (0, 255, 0)
    // C = (0, 0, 255)

    // Input:    [A
    //            B
    //            C]
    //
    // Expected: [A
    //            A
    //            B
    //            B
    //            C]

    // Arrange
    const HostImage input = test::makeRgbImage(
        1, 3,
        {
            255,   0,    0,  // A
            0,    255,   0,  // B
            0,     0,   255  // C
        }
    );

    constexpr int outputWidth = 1;
    constexpr int outputHeight = 5;

    const HostImage expected = test::makeRgbImage(
        outputWidth, outputHeight,
        {
            255,   0,    0,  // A
            255,   0,    0,  // A
            0,    255,   0,  // B
            0,    255,   0,  // B
            0,     0,   255  // C
        }
    );

    // Act
    const HostImage actual = test::runResize(input, outputWidth, outputHeight, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeTaller"
    );
}

void testResizeSmaller()
{
    // Shrink [A B C D E] from 5 x 1 to [A B D] at 3 x 1.

    // Arrange
    // TODO: Choose five distinct colors and write the expected three pixels by hand.

    // Act
    // TODO: Resize to 3, 1.

    // Assert
    // TODO: Compare the actual image with the expected image.
}

void testResizeSameDimensions()
{
    // Resizing a 3 x 2 image to its original shape preserves all bytes.

    // Arrange
    // TODO: Use distinct colors in both rows; expected is a copy of input.

    // Act
    // TODO: Resize to the input width and height.

    // Assert
    // TODO: Compare the complete images.
}

void testResizeBothDimensions()
{
    // Resize 2 x 2 to 4 x 4: AB / CD becomes AABB / AABB / CCDD / CCDD.

    // Arrange
    // TODO: Build input and expected images by hand with four distinct colors.

    // Act
    // TODO: Resize to 4, 4.

    // Assert
    // TODO: Compare every pixel to catch row/column indexing mistakes.
}

void testResizeOneInputPixel()
{
    // Enlarge one RGB pixel to 17 x 19, crossing partial 16 x 16 blocks.

    // Arrange
    // TODO: Create a 1 x 1 input and a solid 17 x 19 expected image of the same color.

    // Act
    // TODO: Resize to 17, 19.

    // Assert
    // TODO: Every output pixel must have the original color.
}

void testResizeOneOutputPixel()
{
    // Shrink a distinct-color 3 x 2 image to 1 x 1.

    // Arrange
    // TODO: Expected is the top-left input pixel under our floor-based mapping.

    // Act
    // TODO: Resize to 1, 1.

    // Assert
    // TODO: Compare the output shape and selected RGB value.
}

void testResizeRejectsNullPointers()
{
    // Reject null input and null output pointers independently.

    // Arrange
    // TODO: Create valid separate DeviceImages and a CudaStream; get their views.

    // Act
    // TODO: Set one view's data pointer to nullptr and call launchResize directly.

    // Assert
    // TODO: Catch std::invalid_argument and require rejection; repeat for the other pointer.
}

void testResizeRejectsInvalidDimensions()
{
    // Reject zero and negative width/height on either view.

    // Arrange
    // TODO: Start each case with valid views of separate device allocations.

    // Act
    // TODO: Change one dimension to 0 or -1, then call launchResize directly.

    // Assert
    // TODO: Require std::invalid_argument for each of the eight cases.
}

void testResizeRejectsNonRgbImages()
{
    // Reject non-RGB input or output independently.

    // Arrange
    // TODO: Create valid separate device buffers; modify only the channel metadata.

    // Act
    // TODO: Try channels 1 and 4 on each view using launchResize directly.

    // Assert
    // TODO: Require std::invalid_argument for every case.
}

void testResizeRejectsSameStorage()
{
    // Reject input and output views that use the same device storage.

    // Arrange
    // TODO: Create one DeviceImage; obtain const input and mutable output views.

    // Act
    // TODO: Call launchResize directly with these views.

    // Assert
    // TODO: Require std::invalid_argument.
}

// Integration checklist (belongs in test_pipeline.cu after standalone tests):
// TODO: Allocate a third device buffer at the destination dimensions.
// TODO: Run resize last and download into a destination-sized HostImage.
// TODO: Test resize alone with grayscale and blur explicitly disabled.
// TODO: Test ordering with odd/even counts of preceding filters.
// TODO: Test repeated calls and changing input sizes with a fixed target size.
// TODO: Test invalid enabled resize dimensions and disabled resize behavior.
// TODO: Build and run the full suite after integration.

int main()
{
    try
    {
        test::Runner runner;

        runner.run("Resize bigger horizontal", testResizeBigger);
        // runner.run("testResizeTaller", testResizeTaller);
        // runner.run("testResizeSmaller", testResizeSmaller);
        // runner.run("testResizeSameDimensions", testResizeSameDimensions);
        // runner.run("testResizeBothDimensions", testResizeBothDimensions);
        // runner.run("testResizeOneInputPixel", testResizeOneInputPixel);
        // runner.run("testResizeOneOutputPixel", testResizeOneOutputPixel);
        // runner.run("testResizeRejectsNullPointers", testResizeRejectsNullPointers);
        // runner.run("testResizeRejectsInvalidDimensions", testResizeRejectsInvalidDimensions);
        // runner.run("testResizeRejectsNonRgbImages", testResizeRejectsNonRgbImages);
        // runner.run("testResizeRejectsSameStorage", testResizeRejectsSameStorage);

        // Remove this reminder once all outlines are implemented and registered.
        std::cout << "NOTE: Commented runner entries are unfinished exercises, not passing tests.\n";
        runner.summary("Resize");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Resize test failed: " << error.what() << '\n';
        return 1;
    }
}
