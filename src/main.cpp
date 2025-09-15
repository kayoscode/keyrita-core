#include "keyrita_core/State.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = uint32_t;
constexpr size_t DIM = 50000;
constexpr size_t SIZE = (size_t)DIM * DIM;

int main()
{
   Timer t;
   volatile size_t sum = 0;

   // std::array<mat_t, SIZE>* pValues = new std::array<mat_t, SIZE>();
   // std::span<mat_t, SIZE> values = *pValues;

   // t.Reset();
   // for (size_t i = 0; i < DIM; i++)
   // {
   //    for (size_t j = 0; j < DIM; j++)
   //    {
   //       size_t flatIdx = ComputeFlatIndex<DIM, DIM>(i, j);
   //       values[flatIdx] = flatIdx;
   //       sum += values[flatIdx];
   //    }
   // }

   // std::cout << t.Milliseconds() << "\n";
   // std::cout << sum << "\n";
   // std::cout << values[100] << "\n";

   HeapMatrixState<mat_t, DIM, DIM> matrix;

   t.Reset();
   sum = 0;
   size_t result = matrix.Ops(1,
        Map(matrix, [](mat_t& value, mat_t, size_t flatIdx)
        {
            value = flatIdx;
        }),
        Fold(sum, [](volatile size_t& acc, mat_t value)
        {
            acc += value;
        }));

   std::cout << t.Milliseconds() << "\n";
   std::cout << sum << "\n";
   std::cout << matrix[100] << "\n";
}
