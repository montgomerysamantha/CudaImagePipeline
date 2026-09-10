#include "filters/heart_bokeh.hpp"
#include "reference/heart_bokeh_cpu.hpp"
#include "tests/test_utils.hpp"

// Suggested test order:
//
// TODO: A black image remains black.
// TODO: One bright center pixel produces the 29 x 14 heart shown in the source.
// TODO: A pixel below the threshold does not produce a heart.
// TODO: A colored highlight keeps the expected color.
// TODO: A heart at an image edge is clipped safely.
// TODO: CUDA output matches the CPU reference on a 17 x 19 image.
// TODO: Invalid image shapes are rejected.
// TODO: Invalid intensity values are rejected.

// Add main() after the first test is implemented. Keep this file out of CMake
// until it contains a runnable test program.
