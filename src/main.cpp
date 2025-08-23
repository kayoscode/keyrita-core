#include "keyrita_core/State.h"
#include "Timer.h"

#include <iostream>

using namespace kc;

int main()
{
   size_t sum = 0;
   Timer t;
   MatrixState<size_t, 30, 30, 30, 30, 30, 30> matrix(10);
   std::cout << t.Milliseconds() << "\n";

   t.Reset();

   matrix.Map([&](size_t& value, size_t flatIndex)
   {
      value = flatIndex;
   });

   std::cout << t.Milliseconds() << "\n";
   t.Reset();

   matrix.ForEach([&sum](size_t value) 
   {
      sum += value;
   });

   std::cout << t.Milliseconds() << "\n";
   std::cout << sum << "\n";
}
