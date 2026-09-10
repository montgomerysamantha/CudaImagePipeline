#include "core/image.hpp"
#include "pipeline/image_pipeline.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    const std::string inputPath = argc > 1 ? argv[1] : "assets/lena.jpg";
    const std::string outputPath = argc > 2 ? argv[2] : "output/lena_pipeline.png";

    try
    {
        const HostImage input = loadRgbImage(inputPath);

        PipelineOptions options;
        options.grayscale = true;
        options.gaussianBlur = true;
        options.edgeDetection = true;

        ImagePipeline pipeline(options);
        PipelineTimings timings;
        const HostImage output = pipeline.process(input, &timings);

        const std::filesystem::path destination(outputPath);
        if (destination.has_parent_path())
        {
            std::filesystem::create_directories(destination.parent_path());
        }
        savePngImage(outputPath, output);

        std::cout << "Processed " << input.width << 'x' << input.height << " RGB image\n"
                  << "Host to device: " << timings.uploadMs << " ms\n"
                  << "GPU processing:  " << timings.processingMs << " ms\n"
                  << "Device to host: " << timings.downloadMs << " ms\n"
                  << "Saved: " << outputPath << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Pipeline failed: " << error.what() << '\n';
        return 1;
    }
}

