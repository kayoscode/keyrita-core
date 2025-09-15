#include "keyrita_core/State.hpp"
#include "keyrita_core/State/MatrixState.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = uint32_t;
constexpr size_t SIZE = (size_t)50000 * 50000;

int main()
{
   Timer t;
   size_t sum = 0;

   HeapMatrixState<mat_t, 5000, 5000> matrix;

   t.Reset();
   sum = 0;
   size_t result = matrix.Ops(1,
        Map(matrix, [](mat_t& value, mat_t, size_t flatIdx)
        {
            value = flatIdx;
        }),
        Fold(sum, [](size_t& acc, mat_t value)
        {
            acc += value;
        }));

   std::cout << t.Milliseconds() << "\n";
   std::cout << result << "\n";
   std::cout << matrix[100] << "\n";

   // std::array<mat_t, SIZE>* pValues = new std::array<mat_t, SIZE>();
   // std::span<mat_t, SIZE> values = *pValues;

   // t.Reset();
   // for (size_t i = 0; i < SIZE; i++)
   // {
   //    values[i] = i;
   // }

   // sum = 0;
   // for (size_t i = 0; i < SIZE; i++)
   // {
   //    sum += values[i];
   // }

   // std::cout << t.Milliseconds() << "\n";
   // std::cout << sum << "\n";
   // std::cout << values[100] << "\n";
}
