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

template <typename TExec>
concept MatrixFuncExHasResult = requires(TExec& exec) {
   { exec.GetResult() };
} && (!std::same_as<decltype(std::declval<TExec>().GetResult()), void>);

// Determines if a matrix result has the same type as given TResult (usually the input.)
template <typename TResult, typename TExec>
concept MatrixFuncHasSameResult = requires(TExec& exec) {
   { exec.GetResult() } -> std::convertible_to<TResult>;
};

/**
 * @brief      Traverses the matrix provindg a list of all indices per callback.
 *
 * @tparam     T           The type used
 * @tparam     TTotalSize  The flat size of the matrix
 * @tparam     TDims...    The list of dimensions and sizes known at compile time.
 */
template <size_t... TDims> class MatrixStaticWalker
{
public:
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

template <typename TFunc> class ForEachEx
{
public:
   /**
    * @brief      Constructor: Create or inject state here.
    */
   ForEachEx(TFunc&& func) : mFunc(std::forward<TFunc>(func))
   {
   }

   /**
    * @brief      Callback from the executor, here you get the value, and the indices.
    * Pass along to the client and do whatever you need with the result.
    */
   template <typename T, typename... TIdx>
   inline constexpr void Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      CallClient(value, flatIndex, indices...);
   }

private:
   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(value, indices...);
   }

   const std::decay_t<TFunc> mFunc;
};

template <typename TFunc> class CountIfEx
{
public:
   CountIfEx(TFunc&& pred) : mPred(std::forward<TFunc>(pred)), mCount(0)
   {
   }

   template <typename T, typename... TIdx>
   inline constexpr void Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      if (CallClient(value, flatIndex, indices...))
      {
         mCount++;
      }
   }

   size_t GetResult() const
   {
      return mCount;
   }

private:
   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value, indices...);
   }

   const std::decay_t<TFunc> mPred;
   size_t mCount;
};

template <typename TFunc> class AllEx
{
public:
   AllEx(TFunc&& pred) : mPred(std::forward<TFunc>(pred)), mResult(true)
   {
   }

   template <typename T, typename... TIdx>
   inline constexpr bool Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      if (!CallClient(value, flatIndex, indices...))
      {
         mResult = false;
         return false;
      }

      return true;
   }

   size_t GetResult() const
   {
      return mResult;
   }

private:
   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value, indices...);
   }

   const std::decay_t<TFunc> mPred;
   bool mResult;
};

template <typename TFunc> class AnyEx
{
public:
   AnyEx(TFunc&& pred) : mPred(std::forward<TFunc>(pred)), mResult(false)
   {
   }

   template <typename T, typename... TIdx>
   inline constexpr bool Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      if (CallClient(value, flatIndex, indices...))
      {
         mResult = true;
         return false;
      }

      return true;
   }

   size_t GetResult() const 
   {
      return mResult;
   }

private:
   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&>
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, size_t> && (sizeof...(TIdx) > 1)
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, const T&, TIdx...>
   constexpr bool CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      return mPred(value, indices...);
   }

   const std::decay_t<TFunc> mPred;
   bool mResult;
};

template <typename TFoldResult, typename TFunc> class FoldEx
{
public:
   FoldEx(TFoldResult& result, TFunc&& pred) : mFunc(std::forward<TFunc>(pred)), mResult(result)
   {
   }

   template <typename T, typename... TIdx>
   inline constexpr void Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      CallClient(value, flatIndex, indices...);
   }

private:
   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, TFoldResult&, const T&>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResult, value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, TFoldResult&, const T&, size_t> && (sizeof...(TIdx) > 1)
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResult, value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, TFoldResult&, const T&, TIdx...>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResult, value, indices...);
   }

   const std::decay_t<TFunc> mFunc;
   TFoldResult& mResult;
};

template <typename TMatrix>
concept ValidMapTarget = requires(TMatrix& matrix, size_t flatIdx) {
   // Only valid if it has a method that returns a mutable reference
   {
      matrix.GetRef(flatIdx)
   } -> std::same_as<std::add_lvalue_reference_t<typename TMatrix::value_type>>;
};
// TODO ensure correct size

template <typename TMatrix, typename TFunc>
   requires ValidMapTarget<TMatrix>
class MapEx
{
public:
   /**
    * @brief      Constructor: Create or inject state here.
    */
   MapEx(TMatrix& resultMatrix, TFunc&& func)
      : mFunc(std::forward<TFunc>(func)), mResultMatrix(resultMatrix)
   {
   }

   /**
    * @brief      Callback from the executor, here you get the value, and the indices.
    * Pass along to the client and do whatever you need with the result.
    */
   template <typename T, typename... TIdx>
   inline constexpr void Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      CallClient(value, flatIndex, indices...);
   }

   /**
    * @return      Returns the matrix that was writen to by ref.
    */
   TMatrix& GetResult() const
   {
      return mResultMatrix;
   }

private:
   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, typename TMatrix::value_type&, const T&>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(flatIndex), value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, typename TMatrix::value_type&, const T&, size_t> &&
               (sizeof...(TIdx) > 1)
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(flatIndex), value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, typename TMatrix::value_type&, const T&, TIdx...>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(flatIndex), value, indices...);
   }

   const std::decay_t<TFunc> mFunc;
   TMatrix& mResultMatrix;
};

template <typename TMatrix>
concept WalkableMatrix = requires(const std::remove_pointer_t<TMatrix>& matrix) {
   {
      matrix.DimensionAction([](auto...) {})
   };
};

class MatrixFuncExecutor
{
public:
   /**
    * @brief      Runs a function executor.
    *
    * @param[in]  matrixValues  The span of values representing the matrix.
    * @param      ex            The function executor.
    *
    * @tparam     TFuncEx       Expected to have a member of type void Impl(const T& value, size_t
    * flatIndex, TIdx... indices)
    */
   template <typename TMatrix, typename TFuncEx>
      requires WalkableMatrix<TMatrix>
   static constexpr decltype(auto) Run(TMatrix& matrix, TFuncEx&& ex)
   {
      auto matrixValues = matrix.GetValues();
      TMatrix::template ApplyDims<MatrixStaticWalker>::Walk(
         [&matrixValues, &ex](size_t flatIdx, auto&&... indices)
         {
            // Return the result to see if it should be canceled.
            return ex.Impl(matrixValues[flatIdx], flatIdx, indices...);
         });

      if constexpr (MatrixFuncExHasResult<TFuncEx>)
      {
         return ex.GetResult();
      }
   }
};

/**
 * @brief      Passes if the series of ops in the given order can be executed on a matrix.
 *
 * Rules:
 * 1. Each op's runner is executed in the order given. results can only be returned from the last op
 * in the list.
 * 2. Even if your op returns a bool, the operation will not be short circuited if called here.
 *
 * Unenforced rule:
 * The actions are executed one after another in a single loop, All results must be computed
 * based on the assumption that none of the other values in the matrix have been mutated yet.
 */
class MatrixOpsExecutor
{
public:
   template <typename TMatrix, typename... TOps>
   static constexpr decltype(auto) Run(TMatrix& matrix, TOps&&... ops)
   {
      auto matrixValues = matrix.GetValues();
      TMatrix::template ApplyDims<MatrixStaticWalker>::Walk(
         [&matrixValues, &ops...](size_t flatIdx, auto&&... indices) -> void
         {
            // Recursively execute every op in order. This is all compile time.
            ExecuteNextOp<TOps...>(matrixValues, std::forward<TOps>(ops)..., flatIdx, indices...);
         });

      return ReturnLastOp(ops...);
   }

private:
   template <typename TCurrentOp, typename... TRemainingOps>
   static constexpr void ExecuteNextOp(auto& matrixValues, TCurrentOp&& currentOp,
      TRemainingOps&&... remainingOps, size_t flatIndex, auto... indices)
   {
      currentOp.Impl(matrixValues[flatIndex], flatIndex, indices...);
      ExecuteNextOp<TRemainingOps...>(
         matrixValues, std::forward<TRemainingOps>(remainingOps)..., flatIndex, indices...);
   }

   template <typename TCurrentOp>
   static constexpr void ExecuteNextOp(
      auto& matrixValues, TCurrentOp&& currentOp, size_t flatIndex, auto... indices)
   {
      currentOp.Impl(matrixValues[flatIndex], flatIndex, indices...);
   }

   template <typename TCurrentOp, typename... TRemainingOps>
   static constexpr decltype(auto) ReturnLastOp(TCurrentOp&& currentOp, TRemainingOps&&... remainingOps)
   {
      return ReturnLastOp(remainingOps...);
   }

   template <typename TCurrentOp> static constexpr decltype(auto) ReturnLastOp(TCurrentOp&& currentOp)
   {
      if constexpr (MatrixFuncExHasResult<TCurrentOp>)
      {
         return currentOp.GetResult();
      }
   }
};
}   // namespace kc
