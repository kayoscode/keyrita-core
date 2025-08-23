#include "keyrita_core/State.h"

#include <iostream>

using namespace kc;

int main()
{
   MatrixState<size_t, 30, 30, 30> matrix(10);
   IMatrixState<size_t, 30, 30, 30>& roMatrix = matrix;

   matrix.Map(
      [&](size_t& value, size_t flatIdx)
      {
         value = flatIdx;
      });

   size_t count = matrix.CountIf([&matrix](size_t value, size_t x, size_t y, size_t z)
   {
      return value == matrix.ToFlatIndex(x, y, z);
   });

   std::cout << count << "\n";
}
