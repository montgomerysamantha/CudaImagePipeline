# CUDA Image Pipeline

**A C++17/CUDA image-processing pipeline that keeps images on the GPU while applying configurable filters.**

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)](https://isocpp.org/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CMake 3.22+](https://img.shields.io/badge/CMake-3.22%2B-064F8C?logo=cmake)](https://cmake.org/)

## Overview

CUDA Image Pipeline applies image filters on an NVIDIA GPU and measures the results. It is built as a learning and performance-engineering project: the output is visual, but the implementation also includes CPU references, CUDA tests, reusable device buffers, and benchmarks that separate processing time from data-transfer overhead.

The pipeline supports:

- Grayscale conversion
- Gaussian blur
- Heart-shaped bokeh
- Sobel edge detection
- Sharpening
- Nearest-neighbor resize

Intermediate images stay in GPU memory while the pipeline runs, so a multi-stage chain can upload once and download once instead of transferring the image around every stage.

## Results at a glance

Recorded on an NVIDIA GTX 1060 3GB with CUDA 12.9 and a Release build:

- Heart bokeh: **113.7× faster end-to-end** than the single-threaded CPU reference
- Heart bokeh: **2.88× faster** with the shared-memory implementation than the global-memory baseline
- Gaussian blur: **68.8× faster in kernel processing** and **6.0× faster end-to-end**
- Complete resident-GPU pipeline: about **4.0–4.4× faster end-to-end** than transferring between every stage
- Correctness: **58 test functions across 9 test programs**, including exact CPU/GPU comparisons

These are measurements from one machine and workload, not universal performance guarantees. See [the technical report](docs/TECHNICAL.md) for methodology, raw results, and limitations.

## Gallery

The gallery shows each effect independently rather than stacking every filter together.

| Filter | Example |
|---|---|
| Grayscale | [Before](assets/kodim01.png) · [After](docs/images/kodim01-grayscale.png) |
| Gaussian blur | [Before](docs/images/gaussian-detail-before.png) · [After](docs/images/gaussian-detail-after.png) |
| Sobel edges | [Before](assets/kodim01.png) · [After](docs/images/kodim01-sobel.png) |
| Heart bokeh | [Before](docs/images/stars-original.png) · [After](docs/images/stars-heart-bokeh.png) |
| Sharpen | [Before](assets/kodim01.png) · [After](docs/images/kodim01-sharpen.png) |

## Build and run

The verified setup is Windows with Visual Studio 2022 Desktop C++, CUDA 12.9, CMake 3.22+, and an NVIDIA GPU. The CMake project is not intentionally Windows-only; Linux verification is planned.

The default architecture is `61`, matching the tested GTX 1060. Set `CUDA_ARCHITECTURES` for another GPU:

```powershell
cmake -S . -B build -DCUDA_ARCHITECTURES="75;86" -DBUILD_TESTING=ON
cmake --build build --config Release