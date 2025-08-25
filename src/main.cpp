#include "keyrita_core/State.h"

#include <iostream>
#include <Timer.h>

using namespace kc;

typedef int mat_t;

int main()
{
   MatrixState<mat_t, 3, 3, 3> matrix(10);

   std::cout << matrix.IsAtDefault() << "\n";

   Timer t;
   matrix.SetValues(0);

   size_t flatIdx;
   size_t x;
   size_t y;
   size_t z;
   bool result = matrix.FindIf([&matrix](mat_t value, size_t x, size_t y, size_t z)
   {
      return value == 0;
   }, x, y, z);

   std::cout << result << "\n";
}
