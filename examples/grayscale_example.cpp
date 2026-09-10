#include "core/image.hpp"
#include "pipeline/image_pipeline.hpp"

// Minimal example body to copy into an executable target when desired.
HostImage runGrayscaleExample(const HostImage& input)
{
    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = false;
    return ImagePipeline(options).process(input);
}

