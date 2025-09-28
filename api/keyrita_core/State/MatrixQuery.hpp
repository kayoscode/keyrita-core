#pragma once

#include "keyrita_core/State/StateBase.hpp"

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
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

// Returns the value type for the given matrix.
template <typename TMatrix>
using MatrixValueType = typename std::remove_cvref_t<TMatrix>::value_type;

// Concept for an action done over the entire matrix at once.
template <typename T, typename TFunc>
concept MatrixBulkAction =
   ScalarStateValue<T> && requires(TFunc predicate, std::span<T> values, size_t count) {
      { predicate(values, count) } -> std::same_as<void>;
   };

template <typename TExec> using MatrixFuncExResult = decltype(std::declval<TExec>().GetResult());

template <typename TExec>
concept MatrixFuncExHasResult = requires(TExec& exec) {
   { exec.GetResult() };
} && (!std::same_as<MatrixFuncExResult<TExec>, void>);

template <typename TMatrix>
concept WalkableMatrix = requires(const TMatrix& matrix) {
   // Make sure there's a method to get access to a readonly span of data.
   { matrix.GetValues() } -> std::convertible_to<std::span<const MatrixValueType<TMatrix>>>;

   // TODO: Please enforce here it has a HasSameDims method
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
   static constexpr bool WalkImpl(TFunc&& func, std::array<size_t, sizeof...(TDims)>& indices)
   {
      for (size_t i = 0; i < TFirstDim; i++)
      {
         indices[TCurrentDimIdx] = i;
         if constexpr (sizeof...(TRemainingDims) > 0)
         {
            // Recursively call and generate the next dimension's index.
            if (!WalkImpl<TFunc, TCurrentDimIdx + 1, TRemainingDims...>(
                   std::forward<TFunc>(func), indices))
            {
               // If our child dimension canceled, we have to cancel here too.
               return false;
            }
         }
         else
         {
            bool canceled = false;

            std::apply(
               [func = std::forward<TFunc>(func), &canceled](auto... idx)
               {
                  // Check for cancel if applicable.
                  if constexpr (std::is_convertible_v<
                                   std::invoke_result_t<TFunc, size_t, decltype(idx)...>, bool>)
                  {
                     if (!func(ComputeFlatIndex<TDims...>(idx...), idx...))
                     {
                        canceled = true;
                     }
                  }
                  else
                  {
                     func(ComputeFlatIndex<TDims...>(idx...), idx...);
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
      }

      // No cancel if we iterate through normal.
      return true;
   }
};

template <typename TFunc> class ForEach
{
public:
   /**
    * @brief      Constructor: Create or inject state here.
    */
   ForEach(TFunc&& func) : mFunc(std::forward<TFunc>(func))
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

template <typename TFunc> class CountIf
{
public:
   CountIf(TFunc&& pred) : mPred(std::forward<TFunc>(pred)), mCount(0)
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

template <typename TFunc> class All
{
public:
   All(TFunc&& pred) : mPred(std::forward<TFunc>(pred)), mResult(true)
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

template <typename TFunc> class Any
{
public:
   Any(TFunc&& pred) : mPred(std::forward<TFunc>(pred)), mResult(false)
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

template <typename TFoldResult, typename TFunc> class Fold
{
public:
   Fold(TFoldResult& result, TFunc&& pred) : mFunc(std::forward<TFunc>(pred)), mResult(result)
   {
   }

   template <typename T, typename... TIdx>
   inline constexpr void Impl(const T& value, size_t flatIndex, TIdx... indices)
   {
      CallClient(value, flatIndex, indices...);
   }

   TFoldResult& GetResult()
   {
      return mResult;
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
   } -> std::same_as<std::add_lvalue_reference_t<MatrixValueType<TMatrix>>>;
};
// TODO ensure correct size

template <typename TMatrix, typename TFunc>
   requires ValidMapTarget<TMatrix>
class Map
{
public:
   /**
    * @brief      Constructor: Create or inject state here.
    */
   Map(TMatrix& resultMatrix, TFunc&& func)
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
      requires std::is_invocable_v<TFunc, MatrixValueType<TMatrix>&, const T&>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(indices...), value);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, MatrixValueType<TMatrix>&, const T&, size_t> &&
               (sizeof...(TIdx) > 1)
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(indices...), value, flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, MatrixValueType<TMatrix>&, const T&, TIdx...>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(indices...), value, indices...);
   }

   const std::decay_t<TFunc> mFunc;
   TMatrix& mResultMatrix;
};

template <typename TMatrix, typename TOtherMatrix, typename TFunc>
   requires ValidMapTarget<TMatrix>
class Zip
{
public:
   /**
    * @brief      Constructor: Create or inject state here.
    */
   Zip(TMatrix& resultMatrix, const TOtherMatrix& otherMatrix, TFunc&& func)
      : mFunc(std::forward<TFunc>(func)), mResultMatrix(resultMatrix),
        mOtherMatrix(otherMatrix)
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
      requires std::is_invocable_v<TFunc, MatrixValueType<TMatrix>&, const T&, const T&>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(indices...), value, mOtherMatrix.GetValue(indices...));
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, MatrixValueType<TMatrix>&, const T&, const T&, size_t> &&
               (sizeof...(TIdx) > 1)
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(indices...), value, mOtherMatrix.GetValue(indices...), flatIndex);
   }

   template <typename T, typename... TIdx>
      requires std::is_invocable_v<TFunc, MatrixValueType<TMatrix>&, const T&, const T&, TIdx...>
   constexpr void CallClient(const T& value, size_t flatIndex, TIdx... indices)
   {
      mFunc(mResultMatrix.GetRef(indices...), value, mOtherMatrix.GetValue(indices...), indices...);
   }

   const std::decay_t<TFunc> mFunc;

   // Matrix data.
   TMatrix& mResultMatrix;
   const TOtherMatrix& mOtherMatrix;
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
   template <WalkableMatrix TMatrix, typename TFuncEx>
   static constexpr decltype(auto) Run(TMatrix& matrix, TFuncEx&& ex)
   {
      auto matrixValues = matrix.GetValues();
      TMatrix::template ApplyDims<MatrixStaticWalker>::Walk(
         [matrixValues, &ex](size_t flatIdx, auto&&... indices)
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
 * @brief Executes a sequence of operations on a walkable matrix, attempting to fuse as many
 *        operations into a single pass over the data as possible.
 *
 * Rules and Behavior:
 * 1. Operations are executed in the order provided. Only the last operation can produce a return
 *    value; intermediate operations are expected to mutate in-place or return void.
 * 2. Even if an operation returns a value (including bool), all operations are executed; there
 *    is no short-circuiting.
 * 3. Loop fusion is attempted: multiple operations are combined into a single traversal of the
 *    matrix whenever possible.
 * 4. If an operation produces a matrix of different dimensions than the current one, a new loop
 *    is started to iterate over the new matrix.
 *
 * Pass Limit Enforcement:
 * - The `maxPasses` parameter specifies the maximum number of passes allowed through the data.
 * - In debug builds, an assert is triggered if the pass count exceeds `maxPasses`.
 * - If `KC_ALWAYS_ENFORCE_PASS_LIMIT` is defined, a runtime exception is thrown in release mode
 *   when the limit is exceeded.
 * - By default, production code will not crash due to excess passes; this enforcement is
 *   primarily for development and testing purposes.
 *
 * Notes:
 * - All operations are assumed to work correctly regardless of the order of mutation within a
 *   single traversal; i.e., operations must not rely on the matrix being partially updated.
 * - The executor relies on matrix spans remaining valid for the duration of all operations.
 * - Compile-time checks ensure that only walkable matrices and valid operations can be executed.
 */
class MatrixOpsExecutor
{
public:
   template <WalkableMatrix TMatrix, typename... TOps>
   static constexpr decltype(auto) Run(int maxPasses, TMatrix& matrix, TOps&&... ops)
   {
      RunImpl(
         maxPasses, matrix,
         [](size_t, auto...)
         {
         },
         std::forward<TOps>(ops)...);
      return ReturnLastOp(std::forward<TOps>(ops)...);
   }

private:
   template <WalkableMatrix TMatrix, typename TCurrentRunner, typename TCurrentOp, typename... TOps>
   static constexpr void RunImpl(int& maxPasses, TMatrix& matrix, TCurrentRunner&& runner,
      TCurrentOp&& currentOp, TOps&&... ops)
   {
      auto matrixValues = matrix.GetValues();

      // If there's no return, pass on the previous matrix to the next op.

      // If there's a return, and it's a walkable matrix:
      // 1. If they're the same size, don't create a new walker and simply pass the result of the
      // previous op to the next.
      // 2. If they are a different size, create a walker, and call the runner and current op, then
      //    create a new runner and continue.
      if constexpr (!MatrixFuncExHasResult<TCurrentOp>)
      {
         RunImpl(
            maxPasses, matrix,
            [runner = std::forward<TCurrentRunner>(runner), &currentOp, matrixValues](
               size_t flatIdx, auto... indices)
            {
               runner(flatIdx, indices...);
               currentOp.Impl(matrixValues[flatIdx], flatIdx, indices...);
            },
            ops...);
      }
      // We require both a matrix result and for that matrix to be a reference to continue
      // iterating.
      else if constexpr (WalkableMatrix<MatrixFuncExResult<TCurrentOp>> &&
                         std::is_reference_v<MatrixFuncExResult<TCurrentOp>>)
      {
         // If the dimensions are the same size, continue chaining the ops together.
         auto& nextInput = currentOp.GetResult();

         if constexpr (TMatrix::template HasSameDims<MatrixFuncExResult<TCurrentOp>>())
         {
            RunImpl(
               maxPasses, nextInput,
               [runner = std::forward<TCurrentRunner>(runner), &currentOp, matrixValues](
                  size_t flatIdx, auto... indices)
               {
                  runner(flatIdx, indices...);
                  currentOp.Impl(matrixValues[flatIdx], flatIdx, indices...);
               },
               ops...);
         }
         else
         {
            // Combine the current runner into one loop by executing here.
            ExecuteRunner(maxPasses, matrix, std::forward<TCurrentRunner>(runner),
               std::forward<TCurrentOp>(currentOp));

            // Start a new loop by creating a new runner and continue with the remaining ops.
            RunImpl(
               maxPasses, nextInput,
               [](size_t, auto...)
               {
               },
               ops...);
         }
      }
      else
      {
         static_assert(
            false, "Return value must be a matrix reference to continue chaining operations.");
      }
   }

   template <WalkableMatrix TMatrix, typename TCurrentRunner, typename TCurrentOp>
   static constexpr void RunImpl(
      int& maxPasses, TMatrix& matrix, TCurrentRunner&& runner, TCurrentOp&& currentOp)
   {
      // No matter what, if there's nothing left, walk the matrix using the runner.
      auto matrixValues = matrix.GetValues();
      ExecuteRunner(maxPasses, matrix, std::forward<TCurrentRunner>(runner),
         std::forward<TCurrentOp>(currentOp));
   }

   template <typename TMatrix, typename TRunner, typename TCurrentOp>
   static constexpr void ExecuteRunner(
      int& maxPasses, TMatrix& matrix, TRunner&& runner, TCurrentOp&& currentOp)
   {
      assert(maxPasses > 0 && "Number of passes through the data exceeded max.");

#if defined(NDEBUG)
      if (maxPasses <= 0)
      {
         throw std::runtime_error("Number of passes through the data exceeded the max.");
      }
#endif

      maxPasses--;

      auto matrixValues = matrix.GetValues();

      TMatrix::template ApplyDims<MatrixStaticWalker>::Walk(
         [matrixValues, runner = std::forward<TRunner>(runner), &currentOp](
            size_t flatIdx, auto&&... indices) -> void
         {
            // Call all the runners setup in the op chain.
            runner(flatIdx, indices...);
            currentOp.Impl(matrixValues[flatIdx], flatIdx, indices...);
         });
   }

   template <typename TCurrentOp, typename... TRemainingOps>
   static constexpr decltype(auto) ReturnLastOp(
      TCurrentOp&& currentOp, TRemainingOps&&... remainingOps)
   {
      return ReturnLastOp(remainingOps...);
   }

   template <typename TCurrentOp>
   static constexpr decltype(auto) ReturnLastOp(TCurrentOp&& currentOp)
   {
      if constexpr (MatrixFuncExHasResult<TCurrentOp>)
      {
         return currentOp.GetResult();
      }
   }
};
}   // namespace kc
