#include "keyrita_core/State.hpp"
#include "keyrita_core/State/MatrixState.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"

#include <Timer.hpp>
#include <iostream>

using namespace kc;
using mat_t = uint32_t;

template <size_t... TSizes>
class Foo
{
public:
    template <template <size_t...> typename Bar>
    using Apply = Bar<TSizes...>;
};

int main()
{
   // HeapMatrixState<mat_t, 5000, 5000> matrix;
   // matrix.SetValues(10);

   // Timer t;
   // size_t sum = 0;
   // bool result = matrix.Ops(
   //      MapEx([](mat_t& value, size_t flatIdx)
   //      {
   //          value = flatIdx;
   //      }),
   //      FoldEx(sum, [](size_t& acc, mat_t value)
   //      {
   //          acc += value;
   //      }),
   //      AnyEx([](mat_t value, size_t flatIdx)
   //      {
   //          if (flatIdx != value)
   //          {
   //             std::cout << flatIdx << "\n";
   //          }
   //          return value != flatIdx;
   //      }
   //      ));

   // std::cout << t.Milliseconds() << "\n";
   // std::cout << sum << "\n";
   // std::cout << result << "\n";

   HeapMatrixState<mat_t, 10, 10> mat;
   HeapMatrixState<double, 10, 10> dmat;

   mat.Map([](mat_t& result, mat_t value, size_t flatIdx)
   {
      result = flatIdx;
   });

   mat.ForEach([](mat_t value, size_t x, size_t y)
   {
      std::cout << value << " " << x << " " << y  << "\n";
   });
}
