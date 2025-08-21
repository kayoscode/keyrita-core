#include "keyrita_core/State.h"
#include <iostream>

using namespace kc;

int main()
{
   MatrixState<uint64_t, 8, 8, 8> matrix(10);
   matrix.SetMatrixValue(100, 1, 2, 3);
   std::cout << matrix.GetMatrixValue(1, 2, 3) << "\n";
}
