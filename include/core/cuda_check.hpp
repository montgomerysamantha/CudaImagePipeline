#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

inline void checkCuda(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result == cudaSuccess)
    {
        return;
    }

    std::ostringstream message;
    message << "CUDA error at " << file << ':' << line
            << " while evaluating " << expression << ": "
            << cudaGetErrorString(result);
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

