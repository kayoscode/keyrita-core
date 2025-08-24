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
   std::cout << t.Milliseconds() << "\n";
   t.Reset();

   std::cout << t.Milliseconds() << "\n";
   t.Reset();

   std::cout << matrix.All([](size_t value, size_t idx)
   {
      return value != idx;
   }) << "\n";

   std::cout << t.Milliseconds() << "\n";
   t.Reset();
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