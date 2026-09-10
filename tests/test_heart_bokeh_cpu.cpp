#include "reference/heart_bokeh_cpu.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool value)
{
    if (!value) throw std::runtime_error("Heart bokeh CPU check failed");
}

HostImage blank(int width, int height)
{
    HostImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(image.view().bytes(), 0);
    return image;
}

int main()
{
    try
    {
        // Rectangular images catch swapped traversal dimensions.
        for (const auto dimensions : {std::pair<int,int>{47,31}, {13,41}, {1,1}})
        {
            HostImage input = blank(dimensions.first, dimensions.second);
            for (std::size_t i = 0; i < input.pixels.size(); ++i)
                input.pixels[i] = static_cast<unsigned char>(i % 256);
            const HostImage& source = input;
            HostImage output = input;
            reference::heartBokeh(source.view(), output.view(), 0, 0);
            require(input.pixels == output.pixels);
        }
        HostImage input = blank(47,31);
        const HostImage& source = input;
        HostImage output = input;
        reference::heartBokeh(source.view(), output.view(), 200, 1);
        require(output.pixels == input.pixels);
        const int center = (15 * input.width + 23) * 3;
        for (int c = 0; c < 3; ++c) input.pixels[center+c] = 255;
        reference::heartBokeh(source.view(), output.view(), 255, 1);
        int lit = 0;
        for (std::size_t i = 0; i < output.pixels.size(); i += 3)
            if (output.pixels[i] == 255) ++lit;
        require(lit == 282);
        require(output.pixels[((15+11)*47+23)*3] == 255); // Bottom tip.
        require(output.pixels[((15-8)*47+23)*3] == 0); // Top notch.
        reference::heartBokeh(source.view(), output.view(), 255, 0.5f);
        require(output.pixels[((15+11)*47+23)*3] == 128);
        require(output.pixels[center] == 255); // Saturation.
        for (int c = 0; c < 3; ++c) input.pixels[center+c] = 199;
        reference::heartBokeh(source.view(), output.view(), 200, 1);
        require(output.pixels == input.pixels);
        bool rejected = false;
        try { reference::heartBokeh(source.view(), input.view(), 200, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
        rejected = false;
        try { reference::heartBokeh(source.view(), output.view(), 200,
            std::numeric_limits<float>::quiet_NaN()); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
        std::cout << "Heart bokeh CPU checks passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
