#include "tests/test_runner.hpp"
#include "filters/gaussian_blur.hpp"
#include "filters/grayscale.hpp"
#include "filters/heart_bokeh.hpp"
#include "filters/edge_detection.hpp"
#include "filters/resize.hpp"
#include "filters/sharpen.hpp"
#include <cmath>
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

void testResizeOnlyPipeline()
{
    // Verify destination allocation and download with no preceding filters.
    // Arrange
    const HostImage input = test::makeRgbImage(
        3, 1,
        {255, 0, 0,  0, 255, 0,  0, 0, 255}
    );
    const HostImage expected = test::makeRgbImage(
        5, 1,
        {255, 0, 0,  255, 0, 0,  0, 255, 0,  0, 255, 0,  0, 0, 255}
    );

    PipelineOptions options;
    options.grayscale = false;
    options.gaussianBlur = false;
    options.resize = true;
    options.outputWidth = 5;
    options.outputHeight = 1;
    ImagePipeline pipeline(options);
    PipelineTimings timings;

    // Act
    const HostImage actual = pipeline.process(input, &timings);

    // Assert
    test::requirePixelsEqual(actual, expected, "testResizeOnlyPipeline");
    test::require(
        std::isfinite(timings.uploadMs) && timings.uploadMs >= 0 &&
        std::isfinite(timings.processingMs) && timings.processingMs >= 0 &&
        std::isfinite(timings.downloadMs) && timings.downloadMs >= 0,
        "Resize pipeline must report finite, nonnegative timings"
    );
}

void testResizeAfterFilters()
{
    // Four or five preceding stages put the final input in A or B.
    // The direct chain verifies that resize runs after every enabled filter.
    // Arrange
    const HostImage input = test::makeCoordinatePatternImage(17, 19);

    for (bool sharpen : {false, true})
    {
        PipelineOptions options;
        options.grayscale = true;
        options.gaussianBlur = true;
        options.heartBokeh = true;
        options.bokehThreshold = 77;
        options.bokehIntensity = 0.1f;
        options.edgeDetection = true;
        options.sharpen = sharpen;
        options.sharpenStrength = 0.5f;
        options.resize = true;
        options.outputWidth = 9;
        options.outputHeight = 23;

        const HostImage gray = test::runFilter(input, launchGrayscale);
        const HostImage blurred = test::runFilter(gray, launchGaussianBlur);
        const HostImage bokeh = test::runFilter(
            blurred,
            [&](ConstImageView in, ImageView out, cudaStream_t stream)
            {
                launchHeartBokeh(in, out, options.bokehThreshold, options.bokehIntensity, stream);
            }
        );
        HostImage filtered = test::runFilter(bokeh, launchEdgeDetection);
        if (sharpen)
        {
            filtered = test::runFilter(
                filtered,
                [&](ConstImageView in, ImageView out, cudaStream_t stream)
                {
                    launchSharpen(in, out, options.sharpenStrength, stream);
                }
            );
        }
        const HostImage expected = test::runResize(filtered, 9, 23, launchResize);
        ImagePipeline pipeline(options);

        // Act
        const HostImage actual = pipeline.process(input);

        // Assert
        test::requirePixelsEqual(
            actual,
            expected,
            sharpen ? "Resize after five filters" : "Resize after four filters"
        );
    }
}

void testResizeRepeatedCalls()
{
    // Alternate different contents at one size to catch stale output buffers.
    // Arrange
    const HostImage first = test::makeCoordinatePatternImage(11, 9);
    const HostImage second = test::makeSolidRgbImage(11, 9, {25, 100, 225});
    PipelineOptions options;
    options.grayscale = false;
    options.gaussianBlur = false;
    options.resize = true;
    options.outputWidth = 17;
    options.outputHeight = 19;
    ImagePipeline pipeline(options);

    for (const HostImage* input : {&first, &second, &first, &second})
    {
        const HostImage expected = test::runResize(*input, 17, 19, launchResize);

        // Act
        const HostImage actual = pipeline.process(*input);

        // Assert
        test::requirePixelsEqual(actual, expected, "testResizeRepeatedCalls");
    }
}

void testResizeChangingInputDimensions()
{
    // Keep the destination fixed while inputs grow, shrink, and match it.
    // Arrange
    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = false;
    options.resize = true;
    options.outputWidth = 17;
    options.outputHeight = 19;
    ImagePipeline pipeline(options);

    for (const auto size : {std::pair<int, int>{3, 2}, {47, 31}, {1, 1}, {17, 19}, {3, 2}})
    {
        const HostImage input = test::makeCoordinatePatternImage(size.first, size.second);
        const HostImage gray = test::runFilter(input, launchGrayscale);
        const HostImage expected = test::runResize(gray, 17, 19, launchResize);

        // Act
        const HostImage actual = pipeline.process(input);

        // Assert
        test::requirePixelsEqual(actual, expected, "testResizeChangingInputDimensions");
    }
}

void testResizeInvalidOutputDimensions()
{
    // Reject invalid targets when resize is enabled, before processing starts.
    for (const auto size : {std::pair<int, int>{0, 5}, {-1, 5}, {5, 0}, {5, -1}})
    {
        // Arrange
        PipelineOptions options;
        options.resize = true;
        options.outputWidth = size.first;
        options.outputHeight = size.second;
        bool rejected = false;

        // Act
        try
        {
            ImagePipeline pipeline(options);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }

        // Assert
        test::require(rejected, "Enabled resize must reject nonpositive target dimensions");
    }
}

void testDisabledResizeIgnoresOutputDimensions()
{
    // Both invalid and valid target sizes must be ignored when resize is off.
    // Arrange
    const HostImage input = test::makeCoordinatePatternImage(3, 2);
    for (const auto size : {std::pair<int, int>{0, -1}, {17, 19}})
    {
        PipelineOptions options;
        options.grayscale = false;
        options.gaussianBlur = false;
        options.resize = false;
        options.outputWidth = size.first;
        options.outputHeight = size.second;
        ImagePipeline pipeline(options);

        // Act
        const HostImage actual = pipeline.process(input);

        // Assert
        test::requirePixelsEqual(actual, input, "testDisabledResizeIgnoresOutputDimensions");
    }
}


int main()
{
    try
    {
        test::Runner runner;
        runner.run("testDisabledStagesPreserveInput", testDisabledStagesPreserveInput);
        runner.run("testBokehPipeline", testBokehPipeline);
        runner.run("testSingleEnabledStageMatchesDirectLauncher", testSingleEnabledStageMatchesDirectLauncher);
        runner.run("testStageOrdering", testStageOrdering);
        runner.run("testRepeatedProcessCalls", testRepeatedProcessCalls);
        runner.run("testBufferReallocationForNewDimensions", testBufferReallocationForNewDimensions);
        runner.run("testInvalidInputIsRejected", testInvalidInputIsRejected);
        runner.run("testResizeOnlyPipeline", testResizeOnlyPipeline);
        runner.run("testResizeAfterFilters", testResizeAfterFilters);
        runner.run("testResizeRepeatedCalls", testResizeRepeatedCalls);
        runner.run("testResizeChangingInputDimensions", testResizeChangingInputDimensions);
        runner.run("testResizeInvalidOutputDimensions", testResizeInvalidOutputDimensions);
        runner.run("testDisabledResizeIgnoresOutputDimensions", testDisabledResizeIgnoresOutputDimensions);
        runner.summary("Pipeline");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Pipeline test failed: " << error.what() << '\n';
        return 1;
    }
}
