#pragma once

#include <iostream>

namespace test
{
// Counts completed test functions, not assertions or loop iterations.
class Runner
{
public:
    template <typename TestFunction>
    void run(const char* name, TestFunction testFunction)
    {
        std::cout << "RUN: " << name << " ... " << std::flush;
        try
        {
            testFunction();
        }
        catch (...)
        {
            std::cout << "FAIL\n";
            throw;
        }
        passed_++;
        std::cout << "PASS\n";
    }

    void summary(const char* suite) const
    {
        std::cout << suite << ": " << passed_ << " tests passed\n";
    }

private:
    int passed_ = 0;
};
}
