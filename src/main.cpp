#include "keyrita_core/State.hpp"
#include "keyrita_core/State/MatrixQuery.hpp"
#include "keyrita_core/State/MatrixState.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = uint32_t;
constexpr size_t DIM = 5000;
constexpr size_t SIZE = (size_t)DIM * DIM;

int main()
{
   Timer t;
   volatile size_t sum = 0;

   HeapMatrixState<mat_t, DIM, DIM> matrix;

   t.Reset();
   sum = 0;
   size_t result = matrix.Ops(1,
        Map(matrix, [](mat_t& value, mat_t, size_t x, size_t y)
        {
            value = ComputeFlatIndex<DIM, DIM>(x, y);
        }),
        Fold(sum, [](volatile size_t& acc, mat_t value)
        {
            acc += value;
        }));

   std::cout << t.Milliseconds() << "\n";
   std::cout << sum << "\n";
   std::cout << matrix[100] << "\n";

   std::array<mat_t, SIZE>* pValues = new std::array<mat_t, SIZE>();
   std::span<mat_t, SIZE> values = *pValues;

   t.Reset();
   for (size_t i = 0; i < DIM; i++)
   {
       for (size_t j = 0; j < DIM; j++)
       {
          values[ComputeFlatIndex<DIM, DIM>(i, j)] = ComputeFlatIndex<DIM, DIM>(i, j);
       }
   }

   sum = 0;
   for (size_t i = 0; i < DIM; i++)
   {
       for (size_t j = 0; j < DIM; j++)
       {
          sum += values[ComputeFlatIndex<DIM, DIM>(i, j)];
       }
   }

   std::cout << t.Milliseconds() << "\n";
   std::cout << sum << "\n";
   std::cout << values[100] << "\n";
}
