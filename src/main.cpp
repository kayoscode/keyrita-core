#include "keyrita_core/State.hpp"
#include "keyrita_core/State/MatrixQuery.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = uint32_t;
constexpr size_t DIM = 20000;
constexpr size_t SIZE = (size_t)DIM * DIM;

int main()
{
   HeapMatrixState<mat_t, DIM, DIM> matrix;
   MatrixUtils::FillSequence(matrix);
   std::cout << matrix[100] << "\n";
   Timer t;
   size_t sum = 0;

   std::array<mat_t, SIZE>* pValues = new std::array<mat_t, SIZE>();
   std::span<mat_t, SIZE> values = *pValues;

   t.Reset();
   sum = 0;
   for (size_t i = 0; i < DIM; i++)
   {
      for (size_t j = 0; j < DIM; j++)
      {
         size_t flatIdx = ComputeFlatIndex<DIM, DIM>(i, j);
         values[flatIdx] = flatIdx;
         values[flatIdx] = values[flatIdx] * values[flatIdx];
         sum += values[flatIdx];
      }
   }

   std::cout << t.Microseconds() << "\n";
   std::cout << "Sum: " << sum << "\n";
   std::cout << "Value: " << values[100] << "\n";

   MatrixUtils::Clear(matrix);
   t.Reset();
   sum = 0;
   matrix.Ops(1, Map(matrix,
      [](mat_t& result, mat_t, size_t flatIdx)
      {
         result = flatIdx;
      }),
      Zip(matrix, matrix,
         [](mat_t& result, mat_t v1, mat_t v2, size_t flatIdx)
         {
            result = v1 * v2;
         }),
      Fold(sum,
         [](size_t& acc, mat_t value)
         {
            acc += value;
         }));

   std::cout << t.Microseconds() << "\n";
   std::cout << "Sum: " << sum << "\n";
   std::cout << "Value: " << matrix[100] << "\n";
}
