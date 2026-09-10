#include "filters/heart_bokeh.hpp"
#include "reference/heart_bokeh_cpu.hpp"
#include "tests/test_utils.hpp"
#include <limits>
#include <iostream>

void compare(const HostImage& input, unsigned char threshold, float intensity)
{
    HostImage expected = input;
    reference::heartBokeh(input.view(), expected.view(), threshold, intensity);
    for (auto launch : {launchHeartBokehGlobal, launchHeartBokehShared, launchHeartBokeh})
    {
    const auto actual = test::runFilter(input,
        [=](ConstImageView source, ImageView destination, cudaStream_t stream)
        { launch(source, destination, threshold, intensity, stream); });
    test::requirePixelsEqual(actual, expected, "CPU/GPU bokeh agreement");
    }
}

int main()
{
    try
    {
        for (const auto size : {std::pair<int,int>{1,1}, {1,37}, {43,1}, {17,19}, {47,31}})
        {
            const auto input = test::makeCoordinatePatternImage(size.first, size.second);
            for (const int threshold : {0, 77, 200, 255})
                for (const float intensity : {0.0f, 0.001f, 0.1f, 0.5f, 1.0f})
                    compare(input, static_cast<unsigned char>(threshold), intensity);
        }
        for (const auto position : {std::pair<int,int>{0,0}, {46,0}, {0,30}, {46,30}, {23,15}})
        {
            auto input = test::makeSolidRgbImage(47,31,{0,0,0});
            test::setRgbPixel(input, position.first, position.second, 255,200,120);
            compare(input, 200, 0.5f);
        }
        compare(test::makeSolidRgbImage(19,17,{0,0,0}), 0, 1);
        compare(test::makeSolidRgbImage(19,17,{255,255,255}), 255, 1);

        // Test argument contracts for both implementations before any launch.
        auto input = test::makeSolidRgbImage(4,3,{0,0,0});
        auto output = input;
        const HostImage& source = input;
        auto reject = [&](ConstImageView a, ImageView b, float intensity)
        {
            bool cpu = false, gpu = false;
            try { reference::heartBokeh(a,b,200,intensity); }
            catch (const std::invalid_argument&) { cpu = true; }
            gpu = true;
            for (auto launch : {launchHeartBokehGlobal, launchHeartBokehShared, launchHeartBokeh})
            {
                bool rejected = false;
                try { launch(a,b,200,intensity,nullptr); }
                catch (const std::invalid_argument&) { rejected = true; }
                gpu = gpu && rejected;
            }
            test::require(cpu && gpu, "Invalid argument must be rejected on CPU and GPU");
        };
        auto bad = source.view(); bad.data = nullptr; reject(bad,output.view(),1);
        auto destination = output.view(); destination.data = nullptr;
        reject(source.view(),destination,1);
        for (int dimension : {0,-1})
        { bad = source.view(); bad.width = dimension; reject(bad,output.view(),1); }
        bad = source.view(); bad.height = 2; reject(bad,output.view(),1);
        bad = source.view(); bad.channels = 1; reject(bad,output.view(),1);
        destination = output.view(); destination.channels = 1;
        reject(source.view(),destination,1);
        for (float intensity : {-1.0f, 1.1f, std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN()})
            reject(source.view(),output.view(),intensity);
        reject(source.view(),input.view(),1);
        // Valid allocated overlapping subviews.
        std::vector<unsigned char> storage(39);
        reject({storage.data(),4,3,3},{storage.data()+3,4,3,3},1);
        std::cout << "Heart bokeh GPU and contract checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    { std::cerr << error.what() << '\n'; return 1; }
}
