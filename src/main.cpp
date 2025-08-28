#include "keyrita_core/State.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = int;

int main()
{
   HeapMatrixState<mat_t, 10> matrix(10);
   // matrix.ForEach([](mat_t value)
   // {
   //    std::cout << value << "\n";
   // });

   matrix.Ops(
      [](mat_t value)
      {
         std::cout << value * 2 << "\n";
      },
      [](mat_t value)
      {
         std::cout << value * 4 << "\n";
      });

   // Timer t;
   // matrix.Map(
   //    [&matrix](mat_t& value, size_t idx)
   //    {
   //       value = idx;
   //    });

   // size_t sum = 0;
   // matrix.Fold(sum, [](size_t& currentSum, mat_t value)
   // {
   //    currentSum += value;
   // });

   // std::cout << t.Milliseconds() << "\n";
   // std::cout << sum << "\n";

   // HeapMatrixState<mat_t, 10> matOps();
}
