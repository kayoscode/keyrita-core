#pragma once

#include "keyrita_core/State/StateBase.hpp"

#include <concepts>
#include <cstddef>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace kc
{
/**
 * @return      Returns the number of dimensions given by a templated list of dims
 */
template <size_t... TDims> constexpr int GetNumDims()
{
   return sizeof...(TDims);
}

/**
 * @return      Computes the total number of flat elements given the dimensions of them atrix
 */
template <size_t TFirstDim, size_t... TRemainingDims> constexpr size_t TotalVecSize()
{
   static_assert(TFirstDim > 0, "Each dimension must be greater than zero");

   if constexpr (sizeof...(TRemainingDims) == 0)
   {
      return TFirstDim;
   }
   // Need the else clause or the compiler will try to evaluate TotalVecSize<> in the base case
   // which is invalid.
   else
   {
      return TFirstDim * TotalVecSize<TRemainingDims...>();
   }
}

/**
 * Concept to verify indices are equiped to index a matrix.
 */
template <size_t TNumDims, typename... TIdx>
concept MatrixIndices = (sizeof...(TIdx) == TNumDims) && (std::convertible_to<TIdx, size_t> && ...);

template <size_t TDim, typename TIdx> constexpr size_t ComputeFlatIndexRecursive(TIdx idx)
{
   return idx;
}

template <size_t TDim, size_t... TRemainingDims, typename TIdx, typename... TRemainingIndices>
constexpr size_t ComputeFlatIndexRecursive(TIdx idx, TRemainingIndices... remainingIndices)
{
   // Process current dimension, then recurse upward with updated stride
   size_t stride = TotalVecSize<TRemainingDims...>();
   return idx * stride + ComputeFlatIndexRecursive<TRemainingDims...>(remainingIndices...);
}

template <size_t... TDims, typename... TIdx>
   requires MatrixIndices<sizeof...(TDims), TIdx...>
constexpr size_t ComputeFlatIndex(TIdx... idx)
{
   return ComputeFlatIndexRecursive<TDims...>(idx...);
}

// Find compile time dimension size impl.
template <size_t TRequestedDim, size_t TCurrentDim, size_t... TRemainingDims>
   requires(TRequestedDim == 0)
constexpr size_t GetDimSizeImpl()
{
   return TCurrentDim;
}

// Compile time GetDimensionImpl
template <size_t TRequestedDim, size_t TCurrentDim, size_t... TRemainingDims>
   requires(TRequestedDim > 0 && sizeof...(TRemainingDims) > 0)
constexpr size_t GetDimSizeImpl()
{
   return TRequestedDim == 0 ? TCurrentDim : GetDimSizeImpl<TRequestedDim - 1, TRemainingDims...>();
}

// Concept for an action done over the entire matrix at once.
template <typename T, typename TFunc>
concept MatrixBulkAction =
   ScalarStateValue<T> && requires(TFunc predicate, std::span<T> values, size_t count) {
      { predicate(values, count) } -> std::same_as<void>;
   };

/**
 * @brief      Concept defining an immutable walk through the array (one call per element)
 *
 * @tparam     T         The type parameter of the matrix T
 * @tparam     TNumDims  The number of dimensions in the matrix
 * @tparam     TFunc     The function called per element
 * @tparam     TIdx      The indices passed to the function.
 */
template <typename T, typename TFunc, size_t TNumDims, typename... TIdx>
concept MatrixImmutableWalkClient =
   MatrixIndices<TNumDims, TIdx...> && requires(TFunc func, const T& value, TIdx... indices) {
      { func(value, indices...) };
   };

/**
 * @brief      Concept defining a mutable walk through any matrix.
 *
 * @tparam     T         The type parameter of the matrix T
 * @tparam     TNumDims  The number of dimensions in the matrix
 * @tparam     TFunc     The function called per element
 * @tparam     TIdx      The indices passed to the function.
 */
template <typename T, typename TFunc, size_t TNumDims, typename... TIdx>
concept MatrixMutableWalkClient =
   MatrixIndices<TNumDims, TIdx...> && requires(TFunc func, T& value, TIdx... indices) {
      { func(value, indices...) };
   };

/**
 * @brief      Concept defining the function parameters required for a fold operation.
 * [](const TFoldType& acc, const T& value, indices...) -> TFoldType
 *
 * @tparam     T         The type parameter of the matrix T
 * @tparam     TNumDims  The number of dimensions in the matrix
 * @tparam     TFunc     The function called per element
 * @tparam     TIdx      The indices passed to the function.
 */
template <typename TFoldType, typename T, typename TFunc, size_t TNumDims, typename... TIdx>
concept MatrixFoldClient =
   MatrixIndices<TNumDims, TIdx...> &&
   requires(TFoldType& initialValue, TFunc func, const T& value, TIdx... indices) {
      { func(initialValue, value, indices...) };
   };

/**
 * @brief      Traverses the matrix provindg a list of all indices per callback.
 *
 * @tparam     T           The type used
 * @tparam     TTotalSize  The flat size of the matrix
 * @tparam     TDims...    The list of dimensions and sizes known at compile time.
 */
template <typename T, size_t... TDims> struct MatrixStaticWalker
{
   /**
    * @brief      Implements the walk operation keeping track of indices for each dimension.
    * Useful for iterating over a single slice.
    *
    * This operation always returns the flat index, followed by a list of the matrix n dimensional
    * indices.
    *
    * @param[in]  matrixValues    The matrix values
    * @param      func            The callback per element [](size_t flatIdx, size_t... indices) {}
    * @param      indices         The indices
    * @param[in]  flatIdx         The flat index
    *
    * @return     True if the operation returned false indicating a cancel.
    */
   template <typename TFunc> static constexpr void Walk(TFunc&& func)
   {
      std::array<size_t, sizeof...(TDims)> dims{};
      WalkImpl<TFunc, 0, TDims...>(std::forward<TFunc>(func), dims);
   }

private:
   template <typename TFunc, size_t TCurrentDimIdx, size_t TFirstDim, size_t... TRemainingDims>
   static constexpr bool WalkImpl(
      TFunc&& func, std::array<size_t, sizeof...(TDims)>& indices, size_t flatIdx = 0)
   {
      // Get the stride
      size_t flatIdxStride = 1;

      if constexpr (sizeof...(TRemainingDims) > 0)
      {
         flatIdxStride = TotalVecSize<TRemainingDims...>();
      }

      for (size_t i = 0; i < TFirstDim; i++)
      {
         indices[TCurrentDimIdx] = i;
         if constexpr (sizeof...(TRemainingDims) > 0)
         {
            // Recursively call and generate the next dimension's index.
            if (!WalkImpl<TFunc, TCurrentDimIdx + 1, TRemainingDims...>(
                   std::forward<TFunc>(func), indices, flatIdx))
            {
               // If our child dimension canceled, we have to cancel here too.
               return false;
            }
         }
         else
         {
            bool canceled = false;

            std::apply(
               [&](auto... idx)
               {
                  // Check for cancel if applicable.
                  if constexpr (std::is_convertible_v<
                                   std::invoke_result_t<TFunc, size_t, decltype(idx)...>, bool>)
                  {
                     if (!func(flatIdx, idx...))
                     {
                        canceled = true;
                     }
                  }
                  else
                  {
                     func(flatIdx, idx...);
                  }
               },
               indices);

            // Return false to execute the cancel here. Note the compiler should optimize this out
            // completely If the function is not cancelable.
            if (canceled)
            {
               return false;
            }
         }

         // Increase the offset by the stride to move to the next dimension.
         flatIdx += flatIdxStride;
      }

      // No cancel if we iterate through normal.
      return true;
   }
};

/**
 * @brief      Implements the foreach functional pattern.
 * [](const T& value, indices...) -> void;
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixForEach
{
public:
   template <typename TFunc>
   static constexpr void Run(std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& f)
   {
      MatrixStaticWalker<T, TDims...>::Walk(GetRunner(matrixValues, std::forward<TFunc>(f)));
   }

   template <typename TFunc>
   static constexpr auto GetRunner(std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      return [matrixValues, predicate = std::forward<TFunc>(predicate)](size_t flatIdx, auto&&... indices)
      {
         return Impl(predicate, matrixValues[flatIdx], flatIdx, indices...);
      };
   }

private:
   template <typename TFunc, typename... TIdx>
   static constexpr void Impl(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      CallClient(std::forward<TFunc>(f), value, flatIndex, indices...);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   static constexpr void CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      f(value);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   static constexpr void CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      f(value, flatIndex);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   static constexpr void CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      f(value, indices...);
   }
};

/**
 * @brief      Implements the All functional pattern. If all selected elements return true for the
 * predicate true is returned, false otherwise.
 * [](const T& value, indices...) -> bool;
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixAllQuery
{
public:
   template <typename TFunc>
   static constexpr bool Run(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      bool result = true;
      MatrixStaticWalker<T, TDims...>::Walk(GetRunner(result, matrixValues, std::forward<TFunc>(predicate)));
      return result;
   }

   template <typename TFunc>
   static constexpr auto GetRunner(bool& result, std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      return [matrixValues, &result, predicate = std::forward<TFunc>(predicate)](size_t flatIdx, auto&&... indices)
      {
         return Impl(predicate, result, matrixValues[flatIdx], flatIdx, indices...);
      };
   }

private:
   template <typename TFunc>
   static constexpr bool Impl(
      TFunc&& predicate, bool& result, const T& value, size_t flatIndex, auto&&... indices)
   {
      if (!CallClient(std::forward<TFunc>(predicate), value, flatIndex, indices...))
      {
         result = false;
         return false;
      }

      return true;
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value, flatIndex);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value, indices...);
   }
};

/**
 * @brief      Implements the All functional pattern. If all selected elements return true for the
 * predicate true is returned, false otherwise.
 * [](const T& value, indices...) -> bool;
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixAnyQuery
{
public:
   template <typename TFunc>
   static constexpr bool Run(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      bool result = false;

      // Return false to break out of the iteration if we fail the condition at any point.
      MatrixStaticWalker<T, TDims...>::Walk(
         [matrixValues, &result, predicate = std::forward<TFunc>(predicate)](
            size_t flatIdx, auto&&... indices)
         {
            return Impl(predicate, result, matrixValues[flatIdx], flatIdx, indices...);
         });

      return result;
   }

private:
   template <typename TFunc>
   static constexpr bool Impl(
      TFunc&& predicate, bool& result, const T& value, size_t flatIndex, auto&&... indices)
   {
      if (CallClient(std::forward<TFunc>(predicate), value, flatIndex, indices...))
      {
         result = true;
         return false;
      }

      return true;
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value, flatIndex);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value, indices...);
   }
};

/**
 * @brief      Implements fold across a set of values in the given matrix.
 * [](const T& acc, const T& value, indices...) -> T;
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixFoldQuery
{
public:
   template <typename TFoldResult, typename TFunc>
   static constexpr void Run(
      TFoldResult& acc, std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& f)
   {
      MatrixStaticWalker<T, TDims...>::Walk(
         [matrixValues, &acc, f = std::forward<TFunc>(f)](size_t flatIndex, auto&&... indices)
         {
            Impl(f, acc, matrixValues[flatIndex], flatIndex, indices...);
         });
   }

private:
   template <typename TFoldResult, typename TFunc>
   static constexpr void Impl(
      TFunc&& f, TFoldResult& acc, const T& value, size_t flatIndex, auto&&... indices)
   {
      CallClient<TFoldResult>(std::forward<TFunc>(f), acc, value, flatIndex, indices...);
   }

   template <typename TFoldResult, typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, TFoldResult&, const T&>
   static constexpr void CallClient(
      TFunc&& f, TFoldResult& acc, const T& value, size_t flatIndex, TIdx... indices)
   {
      f(acc, value);
   }

   template <typename TFoldResult, typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, TFoldResult&, const T&, size_t> && (sizeof...(TIdx) > 1)
   static constexpr void CallClient(
      TFunc&& f, TFoldResult& acc, const T& value, size_t flatIndex, TIdx... indices)
   {
      f(acc, value, flatIndex);
   }

   template <typename TFoldResult, typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, TFoldResult&, const T&, TIdx...>
   static constexpr void CallClient(
      TFunc&& f, TFoldResult& acc, const T& value, size_t flatIndex, TIdx... indices)
   {
      f(acc, value, indices...);
   }
};

/**
 * @brief      Implements the CountIf functional pattern. Each element that passes the predicate
 * increments a counter which is returned.
 * [](const T& value, indices...) -> bool;
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixCountIf
{
public:
   template <typename TFunc>
   static constexpr size_t Run(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      size_t count = 0;
      MatrixStaticWalker<T, TDims...>::Walk(
         [matrixValues, &count, predicate = std::forward<TFunc>(predicate)](
            size_t flatIdx, auto&&... indices)
         {
            return Impl(predicate, count, matrixValues[flatIdx], flatIdx, indices...);
         });

      return count;
   }

private:
   template <typename TFunc>
   static constexpr void Impl(TFunc&& predicate, size_t& count, const T& value, size_t flatIndex, auto&&... indices)
   {
      if (CallClient(std::forward<TFunc>(predicate), value, flatIndex, indices...))
      {
         count++;
      }
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value, flatIndex);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   static constexpr bool CallClient(TFunc&& f, const T& value, size_t flatIndex, TIdx... indices)
   {
      return f(value, indices...);
   }
};

/**
 * @brief      Implements the FindIf functional call. Returns the flat or matrix indices of the
 * first found element matching the predicate.
 * [](const T& value, indices...) -> bool;
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixFindIfQuery
{
public:
   template <typename TFunc, typename... TIdx>
      requires(sizeof...(TIdx) > 0)
   static constexpr bool Run(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate, TIdx&... outIdx)
   {
      bool found = false;

      // Always walk through with the all indices walker. Pass those indices to the predicate if
      // possible, otherwise pass nothing. Compute the flat index if we find the right result and
      // store it in the output indices.

      MatrixStaticWalker<T, TDims...>::Walk(
         [&matrixValues, &found, &predicate, &outIdx...](size_t flatIndex, auto... indices)
         {
            // Query the predicate value.
            if (CallClientHelper(matrixValues[flatIndex], std::forward<TFunc>(predicate), indices...))
            {
               if constexpr (sizeof...(TIdx) == sizeof...(indices))
               {
                  // Pack all the indices.
                  ((outIdx = indices), ...);
               }
               else
               {
                  // Convert to a flat index and set it.
                  PackSingleIndex(outIdx..., ComputeFlatIndex<TDims...>(indices...));
               }

               // Return false to cancel.
               found = true;
               return false;
            }

            return true;
         });

      return found;
   }

private:
   static constexpr void PackSingleIndex(size_t& outIdx, size_t index)
   {
      outIdx = index;
   }

   /**
    * @brief      Call predicate with all indices.
    */
   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   static constexpr bool CallClientHelper(const T& value, TFunc&& predicate, TIdx... indices)
   {
      // Pass all the indices directly.
      return predicate(value, indices...);
   }

   /**
    * @brief      Call predicate with flat index.
    */
   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) != 1)
   static constexpr bool CallClientHelper(const T& value, TFunc&& predicate, TIdx... indices)
   {
      // It takes in a single flat index, so compute that here.
      return predicate(value, ComputeFlatIndex<TDims...>(indices...));
   }

   /**
    * @brief      Call predicate with flat index.
    */
   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   static constexpr bool CallClientHelper(const T& value, TFunc&& predicate, TIdx... indices)
   {
      // It takes in a single flat index, so compute that here.
      return predicate(value);
   }
};

/**
 * @brief      Implements the map operator. Takes in one value and transforms it to another value.
 * [](T& value, indices...)
 *
 * @tparam     T      The type on which to operate.
 * @tparam     TDims  The static dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixMap
{
public:
   template <typename TFunc>
   static constexpr void Run(std::span<T, TotalVecSize<TDims...>()> matrixValues, TFunc&& f)
   {
      MatrixStaticWalker<T, TDims...>::Walk(
         [matrixValues, f = std::forward<TFunc>(f)](size_t flatIndex, auto&&... indices)
         {
            // Discard any cancellation return, the value is set inside the function
            Impl(f, matrixValues[flatIndex], flatIndex, indices...);
         });
   }

private:
   template <typename TFunc> static constexpr void Impl(TFunc&& f, T& value, size_t flatIndex, auto&&... indices)
   {
      CallClient(std::forward<TFunc>(f), value, flatIndex, indices...);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, T&>
   static constexpr void CallClient(TFunc&& f, T& value, size_t flatIndex, TIdx... indices)
   {
      f(value);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, T&, size_t> && (sizeof...(TIdx) > 1)
   static constexpr void CallClient(TFunc&& f, T& value, size_t flatIndex, TIdx... indices)
   {
      f(value, flatIndex);
   }

   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, T&, TIdx...>
   static constexpr void CallClient(TFunc&& f, T& value, size_t flatIndex, TIdx... indices)
   {
      f(value, indices...);
   }
};
}   // namespace kc