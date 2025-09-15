#pragma once

#include "keyrita_core/State/MatrixAlloc.hpp"
#include "keyrita_core/State/MatrixQuery.hpp"
#include "keyrita_core/State/StateBase.hpp"

namespace kc
{
/**
 * @brief      An abstraction on top of vector state providing utilities to handle N dimensions
 */
template <ScalarStateValue T, size_t... TDims> class IMatrixState
{
public:
   // Accessors
   using value_type = T;

   // Apply the dimensions to a type of your choosing.
   template <template <size_t...> typename TTarget> using ApplyDims = TTarget<TDims...>;

   /**
    * @return      Accesses the value at the given index and returns by ref.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   T operator()(TIdx... indices)
   {
      return GetValues()[ToFlatIndex(indices...)];
   }

   /**
    * @return      Accesses the value at the given index and returns by ref.
    */
   T operator[](size_t flatIndex)
   {
      return GetValues()[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given index pack.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...> && (sizeof...(TIdx) > 1)
   T GetValue(TIdx... indices)
   {
      size_t flatIndex = ToFlatIndex(indices...);
      return (*this)[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given flat index.
    */
   T GetValue(size_t flatIndex)
   {
      return (*this)[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given index pack by reference
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...> && (sizeof...(TIdx) > 1)
   const T& GetRef(TIdx... indices) const
   {
      size_t flatIndex = ToFlatIndex(indices...);
      return GetValues()[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given flat index by reference
    */
   const T& GetRef(size_t flatIndex) const
   {
      return GetValues()[flatIndex];
   }

   /**
    * @brief
    * Iterates through each element providing a references to the output matrix. It may be assigned
    * before exiting your lambda. The input matrix is readonly, but you can pass in self to the
    * target matrix.
    *
    * @param      f      Function called per element. There are 3 valid formats:
    * 1. [](const T& out, const T& input)
    * 1. [](const T& out, const T& input, size_t flatIndex)
    * 1. [](const T& out, const T& input, size_t... NIndices)
    */
   template <typename TMatrix, typename TFunc> TMatrix& Map(TMatrix& outMatrix, TFunc&& mapper)
   {
      return MatrixFuncExecutor::Run(*this, kc::Map(outMatrix, std::forward<TFunc>(mapper)));
   }

   /**
    * @brief      Iterates through each element in the matrix and calls your callback.
    *
    * @param      f      Function called per element. There are 3 valid formats:
    * 1. [](const T& value)
    * 2. [](const T& value, size_t flatIndex)
    * 3. [](const T& value, size_t... NIndices)
    */
   template <typename TFunc> void ForEach(TFunc&& f) const
   {
      MatrixFuncExecutor::Run(*this, kc::ForEach(std::forward<TFunc>(f)));
   }

   /**
    * @brief      Counts the number of elements in the matrix that match the given predicate.
    *
    * @param      f      Predicate called per element. There are 3 valid formats:
    * 1. [](const T& value) -> bool
    * 2. [](const T& value, size_t flatIndex) -> bool
    * 3. [](const T& value, size_t... NIndices) -> bool
    *
    * @return     The number of elements that matched the predicate.
    */
   template <typename TFunc> size_t CountIf(TFunc&& pred) const
   {
      return MatrixFuncExecutor::Run(*this, kc::CountIf(std::forward<TFunc>(pred)));
   }

   /**
    * @brief      Returns true if all the elements in the matrix match the predicate
    *
    * @param      f      Predicate called per element. There are 3 valid formats:
    * 1. [](const T& value) -> bool
    * 2. [](const T& value, size_t flatIndex) -> bool
    * 3. [](const T& value, size_t... NIndices) -> bool
    *
    * @return     True if all elements match.
    */
   template <typename TFunc> bool All(TFunc&& pred) const
   {
      return MatrixFuncExecutor::Run(*this, kc::All(std::forward<TFunc>(pred)));
   }

   /**
    * @brief      Returns true if any of the elements in the matrix match the predicate
    *
    * @param      f      Predicate called per element. There are 3 valid formats:
    * 1. [](const T& value) -> bool
    * 2. [](const T& value, size_t flatIndex) -> bool
    * 3. [](const T& value, size_t... NIndices) -> bool
    *
    * @return     True if any elements match.
    */
   template <typename TFunc> bool Any(TFunc&& pred) const
   {
      return MatrixFuncExecutor::Run(*this, kc::Any(std::forward<TFunc>(pred)));
   }

   /**
    * @brief      Iterates over every element in the matrix accumulating a result based on some
    *func. It's notable that the TFoldResult does not have to match the type T.
    *:
    * @param      f      Accumulator called per element. There are 3 valid formats:
    * 1. [](TFoldResult& acc, const T& value)
    * 2. [](TFoldResult& acc, const T& value, size_t flatIndex)
    * 3. [](TFoldResult& acc, const T& value, size_t... NIndices)
    */
   template <typename TFoldResult = T, typename TFunc>
   TFoldResult& Fold(TFoldResult& initialValue, TFunc&& func) const
   {
      return MatrixFuncExecutor::Run(*this, kc::Fold(initialValue, std::forward<TFunc>(func)));
   }

   /**
    * @brief      Returns a readonly view of the data.
    * @return     The readonly data view as a span.
    */
   std::span<const T> GetValues() const
   {
      return mRawData;
   }

   /**
    * @return     The total number of dimensions available for this matrix.
    */
   constexpr int GetNumDims() const
   {
      return sizeof...(TDims);
   }

   /**
    * @return     Returns the size of the given dimension. 0 being the first specified dimension.
    */
   constexpr size_t GetDimSize(size_t dimIndex) const
   {
      if (dimIndex < GetNumDims())
      {
         return mDimSizes[dimIndex];
      }

      return 0;
   }

   /**
    * @brief      Finds the size of the given dimension as a compile time constant. 0 being the
    * first specified dimension
    *
    * @return     The dim size at the given dimension
    */
   template <size_t TDimIndex> size_t constexpr GetDimSize() const
   {
      return GetDimSizeImpl<TDimIndex, TDims...>();
   }

   /**
    * @brief      Calls a lambda passing in the list of dimensions used for this matrix.
    * @tparam     TFunc  The function that gets called with the dimensions
    */
   template <typename TFunc> void DimensionAction(TFunc&& func) const
   {
      std::forward<TFunc>(func)(TDims...);
   }

private:
   template <size_t... TOtherDims> class HasSameDimsChecker
   {
   public:
      constexpr static bool HasSameDims()
      {
         if constexpr (sizeof...(TDims) == (sizeof...(TOtherDims)))
         {
            return ((TDims == TOtherDims) && ...);
         }

         return false;
      }
   };

public:
   /**
    * @return     True if all the dimensions are the same, false otherwise.
    */
   template <typename TMatrix> constexpr bool HasSameDims(const TMatrix& other) const
   {
      return TMatrix::template ApplyDims<HasSameDimsChecker>::HasSameDims();
   }

   /**
    * @return     True if the dims match the dims passed in the parameter pack.
    */
   template <size_t... TOtherDims> constexpr bool HasSameDims() const
   {
      return HasSameDimsChecker<TOtherDims...>::HasSameDims();
   }

   /**
    * @return     Computes the flat index equivalent of the passed index pack.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   constexpr size_t ToFlatIndex(TIdx... indices) const
   {
      return ComputeFlatIndex<TDims...>(indices...);
   }

   /**
    * @return     The total number of elements in this matrix.
    */
   constexpr size_t GetFlatSize() const
   {
      return FlatSize;
   }

protected:
   /**
    * @brief      Sets the readonly memory view for this matrix. This can be changed at runtime if
    * needed.
    */
   void SetReadOnlyData(std::span<const T, TotalVecSize<TDims...>()> data)
   {
      mRawData = std::span<const T>(data);
   }

private:
   // Store a pointer to the raw data here.
   std::span<const T> mRawData;

   // Store the flat size of the array as a const in the base class.
   constexpr static size_t FlatSize = TotalVecSize<TDims...>();

   // Statically for this type stores the dimensions for runtime use.
   static constexpr const int mDimSizes[sizeof...(TDims)]{TDims...};
};

/**
 * @brief      Writable interface to a matrix
 *
 * @tparam     T      value_type for the matrix
 * @tparam     TDims  A size_t list of dimensions
 */
template <template <typename, size_t...> class TAlloc, ScalarStateValue T, size_t... TDims>
   requires MatrixAlloc<TAlloc, T, TotalVecSize<TDims...>()>
class MatrixState : public virtual IMatrixState<T, TDims...>
{
public:
   using allocator_type = TAlloc<T, TDims...>;

   /**
    * @brief      Standard constructor.
    *
    * @param[in]  defaultScalar  The default value that initializes the matrix.
    */
   MatrixState() : mValues(mAllocator.GetVec())
   {
      this->SetReadOnlyData(reinterpret_cast<std::span<const T, FlatSize>&>(mValues));
   }

   /**
    * @return      The value at the given matrix index.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   T& operator()(TIdx... indices)
   {
      return mValues[this->ToFlatIndex(indices...)];
   }

   /**
    * @brief      Performs a series of operations. Attempts to condense all operations into a single
    * loop whenever possible.
    * @param      funcs   The funcs
    * @return     The result of the last expression in the list.
    */
   template <typename... TFuncs> decltype(auto) Ops(TFuncs&&... funcs)
   {
      return MatrixOpsExecutor::Run(*this, std::forward<TFuncs>(funcs)...);
   }

   /**
    * @brief      Sets the value at a given flat index.
    *
    * @param[in]  value      The value
    * @param[in]  flatIndex  The flat index
    */
   virtual void SetValue(const T& value, size_t flatIndex)
   {
      mValues[flatIndex] = value;
   }

   /**
    * @brief      Sets a value at a given matrix pack.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   MatrixState& SetValue(T value, TIdx... indices)
   {
      this->SetValue(value, this->ToFlatIndex(indices...));
      return *this;
   }

   /**
    * @brief      Sets many values and only updates the state and calls the changed callback once.
    *
    * @param[in]  setCallback  The set callback
    *
    * @return     Self
    */
   template <typename TFunc>
      requires MatrixBulkAction<T, TFunc>
   MatrixState& SetValues(TFunc&& setter)
   {
      setter(mValues, FlatSize);
      return *this;
   }

   /**
    * @brief      Sets every value in the matrix to the value
    *
    * @param[in]  value  The value
    *
    * @return     Self
    */
   MatrixState& SetValues(const T& value)
   {
      for (size_t i = 0; i < FlatSize; i++)
      {
         mValues[i] = value;
      }

      return *this;
   }

   /**
    * @return      Returns the value of the matrix at a given index pack by reference
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...> && (sizeof...(TIdx) > 1)
   T& GetRef(TIdx... indices)
   {
      size_t flatIndex = this->ToFlatIndex(indices...);
      return mValues[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given flat index by reference
    */
   T& GetRef(size_t flatIndex)
   {
      return mValues[flatIndex];
   }

   /**
    * @brief
    * Iterates through each element providing a references to the output matrix. It may be assigned
    * before exiting your lambda. The input matrix is readonly, but you can pass in self to the
    * target matrix.
    *
    * @param      f      Function called per element. There are 3 valid formats:
    * 1. [](const T& out, const T& input)
    * 1. [](const T& out, const T& input, size_t flatIndex)
    * 1. [](const T& out, const T& input, size_t... NIndices)
    */
   template <typename TMatrix, typename TFunc> TMatrix& Map(TMatrix& outMatrix, TFunc&& mapper)
   {
      return MatrixFuncExecutor::Run(*this, kc::Map(outMatrix, std::forward<TFunc>(mapper)));
   }

   /**
    * @brief
    * Iterates through each element providing a references to the output matrix. It may be assigned
    * before exiting your lambda. Maps to self.
    *
    * @param      f      Function called per element. There are 3 valid formats:
    * 1. [](const T& out, const T& input)
    * 1. [](const T& out, const T& input, size_t flatIndex)
    * 1. [](const T& out, const T& input, size_t... NIndices)
    */
   template <typename TFunc> MatrixState<TAlloc, T, TDims...>& Map(TFunc&& mapper)
   {
      return MatrixFuncExecutor::Run(*this, kc::Map(*this, std::forward<TFunc>(mapper)));
   }

private:
   constexpr static size_t FlatSize = TotalVecSize<TDims...>();
   TAlloc<T, TDims...> mAllocator;
   std::span<T, FlatSize> mValues;
};

// Vectors

/**
 * @brief      A specialization of a matrix where it's giving one dimension of N size.
 * IVectorState<T, TSize> only provides the readable interface.
 *
 * @tparam     T      A scalar value representing a single element in the vector
 * @tparam     TSize  The length of the vector.
 */
template <ScalarStateValue T, size_t TSize>
class IVectorState : public virtual IMatrixState<T, TSize>
{
public:
};

/**
 * @brief      A specialization of a matrix where it's giving one dimension of N size.
 * VectorState<T, TSize> provides both the read and write interfaces.
 *
 * @tparam     T      A scalar value representing a single element in the vector
 * @tparam     TSize  The length of the vector.
 */
template <template <typename, size_t...> class TAlloc, ScalarStateValue T, size_t TSize>
   requires MatrixAlloc<TAlloc, T, TSize>
class VectorState : public virtual MatrixState<TAlloc, T, TSize>,
                    public virtual IVectorState<T, TSize>
{
public:
   /**
    * @brief      Standard constructor
    *
    * @param[in]  defaultScalar  The default value for every element in the vector state.
    */
   VectorState() : MatrixState<TAlloc, T, TSize>()
   {
   }
};

/**
 * Helper definitions.
 */
template <ScalarStateValue T, size_t... TDims>
class StaticMatrixState : public virtual MatrixState<MatrixStaticAlloc, T, TDims...>
{
public:
   StaticMatrixState() : MatrixState<MatrixStaticAlloc, T, TDims...>()
   {
   }
};

template <ScalarStateValue T, size_t... TDims>
class HeapMatrixState : public virtual MatrixState<MatrixHeapAlloc, T, TDims...>
{
public:
   HeapMatrixState() : MatrixState<MatrixHeapAlloc, T, TDims...>()
   {
   }
};

template <ScalarStateValue T, size_t TSize>
class StaticVectorState : public virtual VectorState<MatrixStaticAlloc, T, TSize>
{
public:
   StaticVectorState() : VectorState<MatrixStaticAlloc, T, TSize>()
   {
   }
};

template <ScalarStateValue T, size_t TSize>
class HeapVectorState : public virtual VectorState<MatrixHeapAlloc, T, TSize>
{
public:
   HeapVectorState() : VectorState<MatrixHeapAlloc, T, TSize>()
   {
   }
};
}   // namespace kc