#include "core/image.hpp"
#include "pipeline/image_pipeline.hpp"

// The unimplemented stages intentionally remain disabled until their TODO kernels
// and, for resize, its differently-sized device allocation have been added.
HostImage runFullPipelineExample(const HostImage& input)
{
    PipelineOptions options;
    options.grayscale = true;
    options.gaussianBlur = true;
    options.edgeDetection = false;
    options.sharpen = false;
    options.resize = false;
    return ImagePipeline(options).process(input);
}

