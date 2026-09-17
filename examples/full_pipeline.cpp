#include "core/image.hpp"
#include "pipeline/image_pipeline.hpp"

// Sharpen is optional. Resize remains disabled until its kernel and
// differently-sized device allocation have been implemented.
HostImage runFullPipelineExample(const HostImage& input)
{
    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = true;
    options.edgeDetection = true;
    options.sharpen = false;
    options.resize = false;
    return ImagePipeline(options).process(input);
}

