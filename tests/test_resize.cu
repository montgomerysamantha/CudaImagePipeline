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

    // A = (255, 0, 0)
    // B = (0, 255, 0)
    // C = (0, 0, 255)
    // D = (204, 45, 191)
    // E = (246, 255, 120)

    // Input:    [A B C D E]
    // Expected: [A B D]

    // Arrange
    constexpr int inputWidth = 5;
    constexpr int inputHeight = 1;
    const HostImage input = test::makeRgbImage(
        inputWidth, inputHeight,
        {
            255,   0,    0,  // A
            0,    255,   0,  // B
            0,     0,   255, // C
            204,  45,   191, // D
            246,  255,  120  // E
        }
    );


    constexpr int outputWidth = 3;
    constexpr int outputHeight = 1;
    const HostImage expected = test::makeRgbImage(
        outputWidth, outputHeight,
        {
            255,   0,    0,  // A
            0,    255,   0,  // B
            204,  45,   191  // D
        }
    );


    // Act
    const HostImage actual = test::runResize(input, outputWidth, outputHeight, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeSmaller"
    );
}

void testResizeSameDimensions()
{
    // Resizing a 3 x 2 image to its original shape preserves all bytes.
    // Arrange
    constexpr int width = 3;
    constexpr int height = 2;
    const HostImage input = test::makeRgbImage(
        width, height,
        {
            255,   0,    0,  // A
            0,    255,   0,  // B
            0,     0,   255, // C
            204,  45,   191, // D
            246,  255,  120, // E
            7,    42,    61  // F
        }
    );

    const HostImage expected = input;

    // Act
    const HostImage actual = test::runResize(input, width, height, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeSameDimensions"
    );
}

void testResizeBothDimensions()
{
    // Resize 2 x 2 to 4 x 4: AB / CD becomes AABB / AABB / CCDD / CCDD.

    // Arrange
    const HostImage input = test::makeRgbImage(
        2, 2,
        {
            255, 0, 0,    0, 255, 0,  // A B
            0, 0, 255,    204, 45, 191 // C D
        }
    );

    const HostImage expected = test::makeRgbImage(
        4, 4,
        {
            255, 0, 0,  255, 0, 0,  0, 255, 0,    0, 255, 0,
            255, 0, 0,  255, 0, 0,  0, 255, 0,    0, 255, 0,
            0, 0, 255,  0, 0, 255,  204, 45, 191, 204, 45, 191,
            0, 0, 255,  0, 0, 255,  204, 45, 191, 204, 45, 191
        }
    );

    // Act
    const HostImage actual = test::runResize(input, 4, 4, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeBothDimensions"
    );
}

void testResizeOneInputPixel()
{
    // Enlarge one RGB pixel to 17 x 19, crossing partial 16 x 16 blocks.

    // Arrange
    constexpr int inputWidth = 1;
    constexpr int inputHeight = 1;

    const HostImage input = test::makeSolidRgbImage(
        inputWidth,
        inputHeight,
        test::RgbPixel{45, 159, 224}
    );

    constexpr int outputWidth = 17;
    constexpr int outputHeight = 19;
    const HostImage expected = test::makeSolidRgbImage(
        outputWidth,
        outputHeight,
        test::RgbPixel{45, 159, 224}
    );

    // Act
    const HostImage actual = test::runResize(input, outputWidth, outputHeight, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeOneInputPixel"
    );
}

void testResizeOneOutputPixel()
{
    // Shrink a distinct-color 3 x 2 image to 1 x 1.

    // Arrange
    constexpr int inputWidth = 3;
    constexpr int inputHeight = 2;
    const HostImage input = test::makeRgbImage(
        inputWidth, inputHeight,
        {
            255,   0,    0,  // A
            0,    255,   0,  // B
            0,     0,   255, // C
            204,  45,   191, // D
            246,  255,  120, // E
            7,    42,    61  // F
        }
    );


    constexpr int outputWidth = 1;
    constexpr int outputHeight = 1;
    const HostImage expected = test::makeRgbImage(
        outputWidth, outputHeight,
        {
            255, 0, 0  // A
        }
    );


    // Act
    const HostImage actual = test::runResize(input, outputWidth, outputHeight, launchResize);

    // Assert
    test::requirePixelsEqual(
        actual,
        expected,
        "testResizeOneOutputPixel"
    );
}

void testResizeRejectsNullInputPointer()
{
    // Reject null input and null output pointers independently.

    // Arrange
    // Create valid separate DeviceImages and a CudaStream; get their views.
    test::CudaStream stream;
    DeviceImage deviceInput(3, 3, 3);
    DeviceImage deviceOutput(3, 3, 3);

    const DeviceImage& readOnlyInput = deviceInput;

    // Views contain metadata and a pointer to the device storage.
    // Changing a view's pointer does not change the owning DeviceImage.
    ConstImageView nullInput = readOnlyInput.view();
    nullInput.data = nullptr;

    bool nullInputWasRejected = false;

    // Act: invalid input, valid output.
    try
    {
        launchResize(
            nullInput,
            deviceOutput.view(),
            stream.get()
        );
    }
    catch (const std::invalid_argument&)
    {
        nullInputWasRejected = true;
    }

    // Assert
    test::require(
        nullInputWasRejected,
        "Resize must reject a null input pointer"
    );
}

void testResizeRejectsNullOutputPointer()
{
    // Reject null input and null output pointers independently.

    // Arrange
    // Create valid separate DeviceImages and a CudaStream; get their views.
    test::CudaStream stream;
    DeviceImage deviceInput(3, 3, 3);
    DeviceImage deviceOutput(3, 3, 3);

    const DeviceImage& readOnlyInput = deviceInput;

    ImageView nullOutput = deviceOutput.view();
    nullOutput.data = nullptr;

    bool nullOutputWasRejected = false;

    // Act: valid input, invalid output.
    try
    {
        launchResize(
            readOnlyInput.view(),
            nullOutput,
            stream.get()
        );
    }
    catch (const std::invalid_argument&)
    {
        nullOutputWasRejected = true;
    }

    // Assert
    test::require(
        nullOutputWasRejected,
        "Resize must reject a null output pointer"
    );
}

void testResizeRejectsInvalidDimensions()
{
    // Reject zero and negative width/height on either view.

    // Arrange
    test::CudaStream stream;
    DeviceImage deviceInput(3, 3, 3);
    DeviceImage deviceOutput(3, 3, 3);
    const DeviceImage& readOnlyInput = deviceInput;

    for (int invalid : {0, -1})
    {
        for (int dimension = 0; dimension < 4; dimension++)
        {
            // Reset views so only one dimension is invalid in each case.
            ConstImageView input = readOnlyInput.view();
            ImageView output = deviceOutput.view();
            int* dimensions[] = {
                &input.width, &input.height, &output.width, &output.height
            };
            *dimensions[dimension] = invalid;
            bool rejected = false;

            // Act: bypass runResize so allocation cannot reject the case first.
            try
            {
                launchResize(input, output, stream.get());
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }

            // Assert
            test::require(
                rejected,
                "Resize must reject invalid dimension " + std::to_string(dimension)
                    + " with value " + std::to_string(invalid)
            );
        }
    }
}

void testResizeRejectsNonRgbImages()
{
    // Reject non-RGB input or output independently.

    // Arrange
    test::CudaStream stream;
    DeviceImage deviceInput(3, 3, 3);
    DeviceImage deviceOutput(3, 3, 3);
    const DeviceImage& readOnlyInput = deviceInput;

    for (int channels : {0, 1, 4})
    {
        for (bool invalidInput : {true, false})
        {
            ConstImageView input = readOnlyInput.view();
            ImageView output = deviceOutput.view();
            if (invalidInput)
            {
                input.channels = channels;
            }
            else
            {
                output.channels = channels;
            }
            bool rejected = false;

            // Act
            try
            {
                launchResize(input, output, stream.get());
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }

            // Assert
            test::require(
                rejected,
                std::string("Resize must reject non-RGB ")
                    + (invalidInput ? "input" : "output")
                    + " with channels " + std::to_string(channels)
            );
        }
    }
}

void testResizeRejectsSameStorage()
{
    // Reject input and output views that use the same device storage.

    // Arrange
    test::CudaStream stream;
    DeviceImage image(3, 3, 3);
    const DeviceImage& readOnlyImage = image;
    bool rejected = false;

    // Act
    try
    {
        launchResize(readOnlyImage.view(), image.view(), stream.get());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    // Assert
    test::require(
        rejected,
        "Resize must reject identical input and output storage"
    );
}


int main()
{
    try
    {
        test::Runner runner;

        runner.run("Resize bigger horizontal", testResizeBigger);
        runner.run("testResizeTaller", testResizeTaller);
        runner.run("testResizeSmaller", testResizeSmaller);
        runner.run("testResizeSameDimensions", testResizeSameDimensions);
        runner.run("testResizeBothDimensions", testResizeBothDimensions);
        runner.run("testResizeOneInputPixel", testResizeOneInputPixel);
        runner.run("testResizeOneOutputPixel", testResizeOneOutputPixel);
        runner.run("testResizeRejectsNullInputPointer", testResizeRejectsNullInputPointer);
        runner.run("testResizeRejectsNullOutputPointer", testResizeRejectsNullOutputPointer);
        runner.run("testResizeRejectsInvalidDimensions", testResizeRejectsInvalidDimensions);
        runner.run("testResizeRejectsNonRgbImages", testResizeRejectsNonRgbImages);
        runner.run("testResizeRejectsSameStorage", testResizeRejectsSameStorage);

        runner.summary("Resize");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Resize test failed: " << error.what() << '\n';
        return 1;
    }
}
