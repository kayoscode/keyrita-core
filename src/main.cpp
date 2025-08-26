#include "keyrita_core/MatrixAlloc.h"
#include "keyrita_core/MatrixState.h"
#include "keyrita_core/MatrixUtils.h"
#include <Timer.h>
#include <iostream>

using namespace kc;
using mat_t = size_t;

int main()
{
   HeapMatrixState<mat_t, 50000, 50000> matrix(10);

   Timer t;
   matrix.Map(
      [](mat_t& value, size_t idx)
      {
         value = idx;
      });

   size_t sum = 0;
   matrix.Fold(sum, [](size_t& currentSum, mat_t value)
   {
      currentSum += value;
   });

   std::cout << t.Milliseconds() << "\n";
   std::cout << sum << "\n";
}
