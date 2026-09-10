#include "filters/gaussian_blur.hpp"
#include "filters/grayscale.hpp"
#include "pipeline/image_pipeline.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

void testDisabledStagesPreserveInput()
{
    const HostImage input = test::makeRgbImage(
        3,
        2,
        {
              1,   2,   3,   4,   5,   6,   7,   8,   9,
             10,  20,  30,  40,  50,  60,  70,  80,  90
        }
    );

    PipelineOptions options;
    options.grayscale = false;
    options.gaussianBlur = false;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;

    ImagePipeline pipeline(options);
    const HostImage output = pipeline.process(input);

    test::requirePixelsEqual(
        output,
        input,
        "testDisabledStagesPreserveInput"
    );
}

void testSingleEnabledStageMatchesDirectLauncher()
{
    // Arrange
    const HostImage input = test::makeRgbImage(
        3,
        2,
        {
              1,   2,   3,   4,   5,   6,   7,   8,   9,
             10,  20,  30,  40,  50,  60,  70,  80,  90
        }
    );

    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = false;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;

    ImagePipeline pipeline(options);

    // Act
    const HostImage pipelineOutput = pipeline.process(input);

    const HostImage directOutput = test::runFilter(input, launchGrayscale);

    // Assert
    test::requirePixelsEqual(
        pipelineOutput,
        directOutput,
        "testSingleEnabledStageMatchesDirectLauncher"
    );
}

void testStageOrdering()
{
    // Arrange
    const HostImage input =
        test::makeCoordinatePatternImage(17, 19);

    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = true;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;

    ImagePipeline pipeline(options);

    // Act
    const HostImage pipelineOutput = pipeline.process(input);

    const HostImage grayscaleOutput =
        test::runFilter(input, launchGrayscale);

    const HostImage directOutput =
        test::runFilter(grayscaleOutput, launchGaussianBlur);

    // Assert
    test::requirePixelsEqual(
        pipelineOutput,
        directOutput,
        "testStageOrdering"
    );
}

void testRepeatedProcessCalls()
{
    // Arrange
    const HostImage firstInput =
        test::makeCoordinatePatternImage(11, 9);

    const HostImage secondInput =
        test::makeSolidRgbImage(11, 9, {25, 100, 225});

    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = true;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;

    ImagePipeline pipeline(options);

    // Act
    const HostImage firstOutput = pipeline.process(firstInput);
    const HostImage repeatedFirstOutput = pipeline.process(firstInput);
    const HostImage secondOutput = pipeline.process(secondInput);
    const HostImage repeatedSecondOutput = pipeline.process(secondInput);

    // Assert
    test::requirePixelsEqual(
        repeatedFirstOutput,
        firstOutput,
        "testRepeatedProcessCalls first input"
    );

    test::requirePixelsEqual(
        repeatedSecondOutput,
        secondOutput,
        "testRepeatedProcessCalls second input"
    );

    test::require(
        firstOutput.pixels != secondOutput.pixels,
        "testRepeatedProcessCalls: different inputs unexpectedly matched"
    );
}

void testBufferReallocationForNewDimensions()
{
    // Arrange
    const HostImage smallInput =
        test::makeCoordinatePatternImage(3, 2);

    const HostImage largeInput =
        test::makeCoordinatePatternImage(19, 17);

    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = false;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;

    ImagePipeline pipeline(options);

    // Act
    const HostImage smallOutput = pipeline.process(smallInput);
    const HostImage largeOutput = pipeline.process(largeInput);

    const HostImage expectedSmallOutput =
        test::runFilter(smallInput, launchGrayscale);

    const HostImage expectedLargeOutput =
        test::runFilter(largeInput, launchGrayscale);

    // Assert
    test::requirePixelsEqual(
        smallOutput,
        expectedSmallOutput,
        "testBufferReallocationForNewDimensions small image"
    );

    test::requirePixelsEqual(
        largeOutput,
        expectedLargeOutput,
        "testBufferReallocationForNewDimensions large image"
    );
}

void testInvalidInputIsRejected()
{
    // Arrange
    PipelineOptions options;
    options.grayscale = false;
    options.gaussianBlur = false;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;

    ImagePipeline pipeline(options);

    const HostImage emptyInput;

    HostImage nonRgbInput;
    nonRgbInput.width = 2;
    nonRgbInput.height = 2;
    nonRgbInput.channels = 1;
    nonRgbInput.pixels = {10, 20, 30, 40};

    bool emptyInputWasRejected = false;
    bool nonRgbInputWasRejected = false;

    // Act
    try
    {
        pipeline.process(emptyInput);
    }
    catch (const std::invalid_argument&)
    {
        emptyInputWasRejected = true;
    }

    try
    {
        pipeline.process(nonRgbInput);
    }
    catch (const std::invalid_argument&)
    {
        nonRgbInputWasRejected = true;
    }

    // Assert
    test::require(
        emptyInputWasRejected,
        "testInvalidInputIsRejected: empty input was accepted"
    );

    test::require(
        nonRgbInputWasRejected,
        "testInvalidInputIsRejected: non-RGB input was accepted"
    );
}

int main()
{
    try
    {
        testDisabledStagesPreserveInput();
        testSingleEnabledStageMatchesDirectLauncher();
        testStageOrdering();
        testRepeatedProcessCalls();
        testBufferReallocationForNewDimensions();
        testInvalidInputIsRejected();
        std::cout << "Pipeline tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Pipeline test failed: " << error.what() << '\n';
        return 1;
    }
}
