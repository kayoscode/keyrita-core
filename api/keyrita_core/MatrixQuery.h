#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace kc
{
template <typename T>
concept ScalarStateValue = std::copyable<T> && std::equality_comparable<T>;

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
 * Matrix walker that iterates through each element in the matrix not maintaining any indices.
 * Can handle both a mutable and immutable walk client.
 */
template <typename T, size_t... TDims> class MatrixWalkerNoIndices
{
public:
   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 0>
   static void WalkReadOnly(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& action)
   {
      Walk<const T>(matrixValues, std::forward<TFunc>(action));
   }

   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 0>
   static void WalkReadWrite(std::span<T, TotalVecSize<TDims...>()> matrixValues, TFunc&& action)
   {
      Walk<T>(matrixValues, std::forward<TFunc>(action));
   }

private:
   template <typename TSpanType, typename TFunc>
   constexpr static void Walk(
      std::span<TSpanType, TotalVecSize<TDims...>()> matrixValues, TFunc&& action)
   {
      for (TSpanType& value : matrixValues)
      {
         if constexpr (std::is_convertible_v<std::invoke_result_t<TFunc, TSpanType&>, bool>)
         {
            // Break out if the action allows it.
            if (!action(value))
            {
               break;
            }
         }
         else
         {
            action(value);
         }
      }
   }
};

/**
 * @brief      Traverses a matrix using a callback which provides a single flat index.
 * Num dims should always be 1 in this case.
 *
 * @tparam     T           The type used
 * @tparam     TTotalSize  The flat size of the matrix
 */
template <typename T, size_t... TDims> class MatrixWalkerFlatIndex
{
public:
   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 1, size_t>
   static void WalkReadOnly(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& action)
   {
      Walk<const T>(matrixValues, std::forward<TFunc>(action));
   }

   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 1, size_t>
   static void WalkReadWrite(std::span<T, TotalVecSize<TDims...>()> matrixValues, TFunc&& action)
   {
      Walk<T>(matrixValues, std::forward<TFunc>(action));
   }

private:
   template <typename TSpanType, typename TFunc>
   constexpr static void Walk(
      std::span<TSpanType, TotalVecSize<TDims...>()> matrixValues, TFunc&& action)
   {
      for (size_t i = 0; i < TotalVecSize<TDims...>(); i++)
      {
         if constexpr (std::is_convertible_v<std::invoke_result_t<TFunc, TSpanType&, size_t>, bool>)
         {
            // Break out if the action allows it.
            if (!action(matrixValues[i], i))
            {
               break;
            }
         }
         else
         {
            action(matrixValues[i], i);
         }
      }
   }
};

/**
 * @brief      Traverses a matrix using a callback which provides n indices, one for each dimension
 *
 * @tparam     T           The type used
 * @tparam     TTotalSize  The flat size of the matrix
 * @tparam     TDims...    The list of dimensions and sizes known at compile time.
 */
template <typename T, size_t... TDims> struct MatrixWalkerMatrixIndices
{
   template <typename TFunc>
   static void WalkReadOnly(std::span<const T, TotalVecSize<TDims...>()> values, TFunc&& func)
   {
      std::array<size_t, sizeof...(TDims)> dims{};
      WalkImpl<const T, TFunc, 0, TDims...>(values, std::forward<TFunc>(func), dims);
   }

   template <typename TFunc>
   static void WalkReadWrite(std::span<T, TotalVecSize<TDims...>()> values, TFunc&& func)
   {
      std::array<size_t, sizeof...(TDims)> dims{};
      WalkImpl<T, TFunc, 0, TDims...>(values, std::forward<TFunc>(func), dims);
   }

private:
   /**
    * @brief      Implements the walk operation keeping track of indices for each dimension.
    * Useful for iterating over a single slice.
    *
    * @param[in]  matrixValues    The matrix values
    * @param      func            The function
    * @param      indices         The indices
    * @param[in]  flatIdx         The flat index
    *
    * @return     True if the operation returned false indicating a cancel.
    */
   template <typename TSpanType, typename TFunc, size_t TCurrentDimIdx, size_t TFirstDim,
      size_t... TRemainingDims>
   static bool WalkImpl(std::span<TSpanType, TotalVecSize<TDims...>()> matrixValues, TFunc&& func,
      std::array<size_t, sizeof...(TDims)>& indices, size_t flatIdx = 0)
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
            if (!WalkImpl<TSpanType, TFunc, TCurrentDimIdx + 1, TRemainingDims...>(
                   matrixValues, std::forward<TFunc>(func), indices, flatIdx))
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
                                   std::invoke_result_t<TFunc, TSpanType&, decltype(idx)...>, bool>)
                  {
                     if (!func(matrixValues[flatIdx], idx...))
                     {
                        canceled = true;
                     }
                  }
                  else
                  {
                     func(matrixValues[flatIdx], idx...);
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
   template <typename TWalker, typename TFunc>
   static constexpr void Impl(std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& f)
   {
      TWalker::WalkReadOnly(matrixValues,
         [f = std::forward<TFunc>(f)](const T& value, auto&&... indices)
         {
            // Here we discard any potential return value from the function to prevent any kind of
            // cancellation.
            f(value, indices...);
         });
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
   template <typename TWalker, typename TFunc>
   static bool Impl(std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      bool result = true;

      // Return false to break out of the iteration if we fail the condition at any point.
      TWalker::WalkReadOnly(matrixValues,
         [&result, predicate = std::forward<TFunc>(predicate)](const T& value, auto&&... indices)
         {
            if (!predicate(value, indices...))
            {
               result = false;
            }

            return result;
         });

      return result;
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
   template <typename TWalker, typename TFunc>
   static bool Impl(std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      bool result = false;

      // Return false to break out of the iteration if we fail the condition at any point.
      TWalker::WalkReadOnly(matrixValues,
         [&result, predicate = std::forward<TFunc>(predicate)](const T& value, auto&&... indices)
         {
            if (predicate(value, indices...))
            {
               result = true;
               return false;
            }

            return true;
         });

      return result;
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
   template <typename TWalker, typename TFoldResult, typename TFunc>
   static void Impl(
      TFoldResult& acc, std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& func)
   {
      TWalker::WalkReadOnly(matrixValues,
         [&acc, func = std::forward<TFunc>(func)](const T& value, auto&&... indices)
         {
            func(acc, value, indices...);
         });
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
   template <typename TWalker, typename TFunc>
   static constexpr size_t Impl(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate)
   {
      size_t acc = 0;
      MatrixFoldQuery<T, TDims...>::template Impl<TWalker, size_t>(acc, matrixValues,
         [predicate = std::forward<TFunc>(predicate)](size_t& acc, const T& value, auto... indices)
         {
            if (predicate(value, indices...))
            {
               acc++;
            }
         });

      return acc;
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
   static constexpr bool Impl(
      std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& predicate, TIdx&... outIdx)
   {
      bool found = false;

      // Always walk through with the all indices walker. Pass those indices to the predicate if
      // possible, otherwise pass nothing. Compute the flat index if we find the right result and
      // store it in the output indices.

      MatrixWalkerMatrixIndices<T, TDims...>::WalkReadOnly(matrixValues,
         [&found, &predicate, &outIdx...](const T& value, auto... indices)
         {
            // Query the predicate value.
            if (CallPredicateHelper(value, std::forward<TFunc>(predicate), indices...))
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

   static constexpr void PackSingleIndex(size_t& outIdx, size_t index)
   {
      outIdx = index;
   }

   /**
    * @brief      Call predicate with all indices.
    */
   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   static constexpr bool CallPredicateHelper(const T& value, TFunc&& predicate, TIdx... indices)
   {
      // Pass all the indices directly.
      return predicate(value, indices...);
   }

   /**
    * @brief      Call predicate with flat index.
    */
   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) != 1)
   static constexpr bool CallPredicateHelper(const T& value, TFunc&& predicate, TIdx... indices)
   {
      // It takes in a single flat index, so compute that here.
      return predicate(value, ComputeFlatIndex<TDims...>(indices...));
   }

   /**
    * @brief      Call predicate with flat index.
    */
   template <typename TFunc, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   static constexpr bool CallPredicateHelper(const T& value, TFunc&& predicate, TIdx... indices)
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
   template <typename TWalker, typename TFunc>
   static constexpr void Impl(std::span<T, TotalVecSize<TDims...>()> matrixValues, TFunc&& f)
   {
      TWalker::WalkReadWrite(matrixValues,
         [f = std::forward<TFunc>(f)](T& value, auto&&... indices)
         {
            // Discard any cancellation return, the value is set inside the function
            f(value, indices...);
         });
   }
};
}   // namespace kc