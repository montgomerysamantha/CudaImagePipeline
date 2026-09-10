#include "pipeline/image_pipeline.hpp"
#include "tests/test_utils.hpp"

#include <exception>
#include <iostream>

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
    // TODO: Enable only grayscale in the pipeline. Compare the pipeline result
    // with a direct call to launchGrayscale using the same input.
}

void testStageOrdering()
{
    // TODO: Run grayscale then Gaussian blur through the pipeline. Run the same
    // launchers manually in that order and compare their final results.
}

void testRepeatedProcessCalls()
{
    // TODO: Call process more than once with the same image and verify stable
    // output. Then repeat with a different image of the same dimensions.
}

void testBufferReallocationForNewDimensions()
{
    // TODO: Process two valid images with different dimensions using one
    // ImagePipeline instance and verify both output shapes.
}

void testInvalidInputIsRejected()
{
    // TODO: Pass an empty or non-RGB HostImage and verify process throws.
}

int main()
{
    try
    {
        testDisabledStagesPreserveInput();
        std::cout << "Pipeline tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Pipeline test failed: " << error.what() << '\n';
        return 1;
    }
}
