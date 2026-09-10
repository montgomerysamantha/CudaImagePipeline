# CUDA image pipeline

This repository now contains a reusable GPU image pipeline alongside the original
standalone CUDA experiments. The working path uploads an RGB image once, runs
grayscale and Gaussian blur using two alternating device buffers, and downloads
the final image once.

Edge detection, sharpen, and resize have launcher interfaces and template source
files but are disabled by default until their kernels are implemented.

## Build with CMake

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Run from the repository root:

```powershell
.\build\Release\cuda_image_pipeline.exe lena.jpg output\lena_pipeline.png
```

Both arguments are optional. The defaults are `lena.jpg` and
`output/lena_pipeline.png`.

## Important files

- `include/core/device_image.hpp`: owns device memory and performs transfers.
- `include/pipeline/image_pipeline.hpp`: configures enabled processing stages.
- `src/pipeline/image_pipeline.cu`: performs the one-copy-in/one-copy-out flow.
- `src/filters/grayscale.cu`: reusable RGB grayscale kernel.
- `src/filters/gaussian_blur.cu`: reusable shared-memory 3x3 Gaussian kernel.
- `src/filters/edge_detection.cu`, `sharpen.cu`, and `resize.cu`: extension templates.

To implement a template stage, replace its `std::logic_error` with a kernel launch
that reads `input`, writes `output`, uses the supplied stream, and checks the launch
with `CUDA_CHECK(cudaGetLastError())`. Then enable that stage in `PipelineOptions`.

Resize additionally needs the pipeline to allocate a destination buffer using
`outputWidth` and `outputHeight`.

