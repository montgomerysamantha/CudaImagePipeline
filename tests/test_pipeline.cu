#include "filters/gaussian_blur.hpp"
#include "filters/grayscale.hpp"
#include "filters/heart_bokeh.hpp"
#include "filters/edge_detection.hpp"
#include "reference/heart_bokeh_cpu.hpp"
#include "pipeline/image_pipeline.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <limits>

void testBokehPipeline()
{
    PipelineOptions options;
    options.grayscale = false;
    options.gaussianBlur = false;
    options.heartBokeh = true;
    options.bokehThreshold = 77;
    options.bokehIntensity = 0.1f;
    ImagePipeline pipeline(options);
    // Repeat inputs and change dimensions to exercise buffer reuse/reallocation.
    for (const auto size : {std::pair<int,int>{17,19}, {17,19}, {47,31}, {1,1}})
    {
        const auto input = test::makeCoordinatePatternImage(size.first,size.second);
        auto expected = input;
        reference::heartBokeh(input.view(),expected.view(),77,0.1f);
        test::requirePixelsEqual(pipeline.process(input),expected,"Pipeline bokeh CPU agreement");
    }
    const auto input = test::makeCoordinatePatternImage(47,31);
    options.grayscale = true;
    options.gaussianBlur = true;
    options.edgeDetection = true;
    const auto gray = test::runFilter(input,launchGrayscale);
    const auto blurred = test::runFilter(gray,launchGaussianBlur);
    const auto bokeh = test::runFilter(blurred,
        [&](ConstImageView a, ImageView b, cudaStream_t s)
        { launchHeartBokeh(a,b,options.bokehThreshold,options.bokehIntensity,s); });
    const auto expected = test::runFilter(bokeh,launchEdgeDetection);
    test::requirePixelsEqual(ImagePipeline(options).process(input),expected,"Bokeh stage ordering");

    options.grayscale = false;
    options.gaussianBlur = false;
    options.edgeDetection = false;
    options.bokehIntensity = 0;
    test::requirePixelsEqual(ImagePipeline(options).process(input),input,"Zero bokeh intensity");
    for (float value : {-1.0f,1.1f,std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity()})
    {
        options.bokehIntensity = value;
        bool rejected = false;
        try { ImagePipeline invalid(options); }
        catch (const std::invalid_argument&) { rejected = true; }
        test::require(rejected,"Invalid enabled bokeh intensity");
    }
    options.heartBokeh = false;
    test::requirePixelsEqual(ImagePipeline(options).process(input),input,"Disabled bokeh ignores settings");
}

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
        testBokehPipeline();
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
