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

## Gaussian Blur Benchmark

This benchmark compares three implementations of a 3×3 Gaussian blur:

- A single-threaded CPU reference implementation
- A naive CUDA kernel that reads neighboring pixels from global memory
- An optimized CUDA kernel that reuses neighboring pixels through shared memory

The benchmark processed a 1960×1960 RGB image over 20 runs. The CPU and GPU outputs matched exactly.

### Test configuration

| Property | Value |
|---|---:|
| Image dimensions | 1960×1960 RGB |
| Image size | 10.991 MiB |
| Runs per measurement | 20 |
| Correctness | Pass |

### Compute-only results

These measurements compare the processing implementations without including CPU-to-GPU or GPU-to-CPU transfers.

| Implementation | Average time |
|---|---:|
| CPU reference | 59.637 ms |
| Naive CUDA kernel | 1.335 ms |
| Shared-memory CUDA kernel | 0.842 ms |

The shared-memory kernel was **1.585× faster** than the naive CUDA kernel. Shared memory helps because threads in the same block can reuse nearby pixels instead of repeatedly fetching them from slower global GPU memory.

Compared with the single-threaded CPU reference, the shared-memory kernel was approximately **70.8× faster at computation alone**. This comparison does not include the time required to move the image between CPU and GPU memory.

### End-to-end GPU results

A complete GPU operation includes uploading the image, executing the kernel, and downloading the result.

```text
CPU memory
    │
    │ Host to device: 4.470 ms
    ▼
GPU memory
    │
    │ Shared Gaussian kernel: 0.842 ms
    ▼
GPU result
    │
    │ Device to host: 4.237 ms
    ▼
CPU memory
```

| Operation | Average time |
|---|---:|
| Host-to-device transfer | 4.470 ms |
| Shared-memory kernel | 0.842 ms |
| Device-to-host transfer | 4.237 ms |
| Measured end-to-end total | 10.209 ms |

Including transfers and other overhead, the GPU completed the full operation **5.841× faster** than the CPU reference.

### Where the GPU time goes

| Portion | Percentage of end-to-end time |
|---|---:|
| Memory transfers | 85.288% |
| Gaussian kernel | 8.248% |
| Other overhead | 6.464% |

The kernel itself accounts for only a small part of the end-to-end GPU time. Most of the time is spent moving the image between CPU and GPU memory.

This is why the image pipeline keeps an image on the GPU while applying multiple filters:

```text
Less efficient for multiple filters:

CPU → GPU → blur → CPU
CPU → GPU → edges → CPU
CPU → GPU → sharpen → CPU


Pipeline approach:

CPU → GPU → grayscale → blur → edges → sharpen → CPU
```

The pipeline pays the upload and download costs once, allowing several GPU operations to share the same transfers. For a single filter, transfer overhead reduces the practical speedup. As more filters are added between the transfers, the cost is spread across more useful GPU work.

These results apply to this image, implementation, build configuration, and test system. Performance may differ with other GPUs, CPUs, image sizes, and compiler settings.

