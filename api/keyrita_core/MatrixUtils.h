#pragma once

#include <concepts>
#include <cpp_events/Event.h>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <wchar.h>

namespace kc
{
/**
 * @brief Perform useful queries and operations on matrices
 */
namespace MatrixUtils
{
template <typename T>
concept SelfAssignable = requires(T& v, const T& other) {
   { v = other } -> std::convertible_to<T>;
};

/**
 * @brief      Concept to check if an addition operation can be done on the type
 */
template <typename T, typename TOut = T>
concept Summable = requires(const T& a, const T& b) {
   { a + b } -> std::convertible_to<TOut>;
} && requires(TOut& v, const TOut& other) {
   { v = other } -> std::convertible_to<TOut>;
};

// General

/**
 * @brief      Sets all values in the matrix to the default value for the type.
 */
template <typename TMatrix, typename T = TMatrix::value_type> void Clear(TMatrix& matrix)
   requires std::is_default_constructible_v<T>
{
   matrix.SetValues(T{});
}

/**
 * @return     Fills each cell in the matrix with each value being set to its iteration order.
 */
template <typename TMatrix> void FillSequence(TMatrix& matrix)
{
   matrix.Map(
      [](auto& value, size_t flatIndex)
      {
         value = flatIndex;
      });
}

/**
 * @return     True if all values in the matrix match a given value.
 */
template <typename TMatrix, typename T> bool AllEqual(const TMatrix& matrix, const T& value)
{
   return matrix.All(
      [&value](const T& inValue)
      {
         return inValue == value;
      });
}

/**
 * @return     True if all values in the matrix match a given value.
 */
template <typename TMatrix, typename T> bool NoneEqual(const TMatrix& matrix, const T& value)
{
   return matrix.Any(
      [&value](const T& inValue)
      {
         return inValue == value;
      });
}

/**
 * @return     True if all values in the matrix match a given value.
 */
template <typename TMatrix, typename T> size_t CountEqual(const TMatrix& matrix, const T& value)
{
   return Count(
      [&value](const T& inValue)
      {
         return inValue == value;
      });
}

// Arithmetic

/**
 * @return     The sum of all elements in the matrix.
 */
template <typename TMatrix, typename T = TMatrix::value_type, typename TOut = T>
   requires Summable<T, TOut> && SelfAssignable<T> && std::is_default_constructible_v<T>
TOut Sum(const TMatrix& matrix)
{
   // Attempt to default construct.
   TOut sum{};

   // Fold over for the sum.
   matrix.template Fold<TOut>(sum,
      [](TOut& acc, const T& value)
      {
         acc = std::move(acc) + value;
      });

   return sum;
}
}   // namespace MatrixUtils
}   // namespace kc
