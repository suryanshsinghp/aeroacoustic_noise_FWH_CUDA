#ifndef TIMER
#define TIMER

#include <chrono>
#include <iostream>
#include <string>

class Timer
{
private:
    std::string func_name;
    std::string unit;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

public:
    Timer(const std::string &f, const std::string &str) // start timer in constructor
        : func_name(f), unit(str), start(std::chrono::high_resolution_clock::now())
    {
    }

    ~Timer()
    { // when out of scope, the destructor print time elapsed
        auto end = std::chrono::high_resolution_clock::now();
        if (unit == "ms")
        {
            std::cout << func_name << " execution time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                      << " ms.\n";
        }
        else if (unit == "s")
        {
            std::cout << func_name << " execution time: "
                      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
                      << " s.\n";
        }
        else if (unit == "µs")
        {
            std::cout << func_name << " execution time: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                      << " µs.\n";
        }
        else
        {
            std::cout << func_name << " execution time: "
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                      << " ns.\n";
        }
    }
};

#endif