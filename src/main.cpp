#include "keyrita_core/State.h"

#include <iostream>
#include <Timer.h>

using namespace kc;

typedef int mat_t;

int main()
{
   MatrixState<mat_t, 3, 3, 3> matrix(10);

   Timer t;
   matrix.Map(
      [&matrix](mat_t& value, size_t idx)
      {
         value = idx;
      });

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

void SyntaxTest()
{
   MatrixState<size_t, 3, 3, 3> matrix(10);

   matrix.Map(
      [&](size_t& value)
      {
         value = 0;
      });

   matrix.Map(
      [&](size_t& value, size_t x, size_t y, size_t z)
      {
         value = 0;
      });

   matrix.Map(
      [&](size_t& value, size_t idx)
      {
         value = idx;
      });

   //Compute the sum using a fold.
   size_t count = matrix.CountIf([](mat_t value)
   {
      return value < 10;
   });

   std::cout << count << "\n";
}