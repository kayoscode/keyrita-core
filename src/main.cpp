#include "keyrita_core/State.h"

#include <iostream>
#include <Timer.h>

using namespace kc;

int main()
{
   MatrixState<size_t, 30, 30, 30> matrix(10);

   Timer t;
   matrix.Map(
      [&](size_t& value, size_t idx)
      {
         value = idx;
      });

   // Compute the sum using a fold.
   size_t count = matrix.Fold(0, [](size_t acc, size_t value)
   {
      acc += value < 40;
      return acc;
   });

   std::cout << count << "\n";
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
}