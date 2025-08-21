#include "keyrita_core/State.h"
#include <iostream>

using namespace kc;

int main()
{
   MatrixState<double, 10, 10> matrix(10);
   std::cout << matrix.GetNumDims() << "\n";

   NestedArrayT<int, 1> nestedArray;
}