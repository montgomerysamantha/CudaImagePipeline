#include "filters/heart_bokeh.hpp"

namespace
{

// Use the same fixed 29 x 14 aperture and anchor (14, 5) as the CPU reference:
//
//     .....######.......######.....
//     ...##########...##########...
//     .#############.#############.
//     #############################
//     #############################
//     #############################
//     .###########################.
//     ...#######################...
//     .....###################.....
//     .......###############.......
//     .........###########.........
//     ...........#######...........
//     .............###.............
//     ..............#..............
//
// TODO: Store the aperture in a form CUDA threads can read efficiently.

// TODO: Add the device-side apertureContains helper.
// TODO: Add the device-side luminance helper.
// TODO: Add a kernel that assigns one thread to each output pixel.
// TODO: Add host-side argument validation.
// TODO: Add the launchHeartBokeh definition and check the kernel launch.

} // namespace
