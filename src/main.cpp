#include "keyrita_core/State.hpp"
#include "keyrita_core/State/MatrixQuery.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = int;

int main()
{
   HeapMatrixState<mat_t, 50000, 50000> matrix(10);

   Timer t;
   size_t sum = 0;
   matrix.Ops(
        MapEx([](mat_t& value, size_t flatIdx)
        {
            value = flatIdx;
        }),
        FoldEx(sum, [](size_t& acc, mat_t value)
        {
            acc += value;
        }));

   std::cout << t.Milliseconds() << "\n";
   std::cout << sum << "\n";
}
