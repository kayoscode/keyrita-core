#pragma once

#include "keyrita_core/MatrixAlloc.h"
#include "keyrita_core/MatrixQuery.h"
#include "keyrita_core/State.h"

namespace kc
{
/**
 * @brief      Class representing a read only view of a matrix state object
 *
 * @tparam     T      The type stored in the matrix.
 * @tparam     TDims  The original dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class IMatrixView
{
public:
   using value_type = T;

   /**
    * @brief      Standard constructor
    * @param[in]  rawData  A raw pointer to readonly memory over which this view will operate.
    */
   IMatrixView(const std::span<const T, TotalVecSize<TDims...>()> rawData) : mRawMatrix(rawData)
   {
   }

private:
   std::span<const T, TotalVecSize<TDims...>()> mRawMatrix;
   std::array<std::tuple<size_t, size_t>, sizeof...(TDims)> mSlice;
};

/**
 * @brief      Class representing a read/write view of a matrix state object
 *
 * @tparam     T      The type stored in the matrix.
 * @tparam     TDims  The original dimensions of the matrix.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixView : public IMatrixView<T, TDims...>
{
public:
};

/**
 * @brief      An abstraction on top of vector state providing utilities to handle N dimensions
 */
template <ScalarStateValue T, size_t... TDims> class IMatrixState : public virtual IReadState
{
public:
   // Accessors
   using value_type = T;

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
      assert(flatIndex < FlatSize);
      return (*this)[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given flat index.
    */
   T GetValue(size_t flatIndex)
   {
      assert(flatIndex < FlatSize);
      return (*this)[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given index pack by reference
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...> && (sizeof...(TIdx) > 1)
   const T& GetRef(TIdx... indices)
   {
      size_t flatIndex = ToFlatIndex(indices...);
      assert(flatIndex < FlatSize);
      return GetValues()[flatIndex];
   }

   /**
    * @return      Returns the value of the matrix at a given flat index by reference
    */
   const T& GetRef(size_t flatIndex)
   {
      assert(flatIndex < FlatSize);
      return GetValues()[flatIndex];
   }

   // ForEach

   template <typename TFunc> void ForEach(TFunc&& f) const
   {
      MatrixForEach<T, TDims...>::Run(GetValues(), std::forward<TFunc>(f));
   }

   // CountIf

   template <typename TFunc> size_t CountIf(TFunc&& f) const
   {
      return MatrixCountIf<T, TDims...>::Run(GetValues(), std::forward<TFunc>(f));
   }

   // All

   template <typename TFunc> bool All(TFunc&& predicate) const
   {
      return MatrixAllQuery<T, TDims...>::Run(GetValues(), std::forward<TFunc>(predicate));
   }

   // Any

   template <typename TFunc> bool Any(TFunc&& predicate) const
   {
      return MatrixAnyQuery<T, TDims...>::Run(GetValues(), std::forward<TFunc>(predicate));
   }

   // Fold

   template <typename TFoldResult = T, typename TFunc>
   void Fold(TFoldResult& initialValue, TFunc&& func) const
   {
      MatrixFoldQuery<T, TDims...>::Run(initialValue, GetValues(), std::forward<TFunc>(func));
   }

   // FindIf

   template <typename TFunc, typename... TIdx>
      requires(sizeof...(TIdx) == sizeof...(TDims) || sizeof...(TIdx) == 1)
   bool FindIf(TFunc&& predicate, TIdx&... indices) const
   {
      return MatrixFindIfQuery<T, TDims...>::Run(
         GetValues(), std::forward<TFunc>(predicate), indices...);
   }

   /**
    * @brief      Returns a readonly view of the data.
    * @return     The readonly data view as a span.
    */
   virtual const std::span<const T, TotalVecSize<TDims...>()> GetValues() const = 0;

   /**
    * @brief      Returns a raw pointer to the underlying flat matrix values.
    *
    * @return     A raw pointer to the underlying data. It is considered unsafe to cast away const
    * and modify the values.
    */
   const std::span<const T> GetValuesUnsized() const
   {
      return GetValues();
   }

   /**
    * @return     The total number of dimensions available for this matrix.
    */
   int GetNumDims() const
   {
      return sizeof...(TDims);
   }

   /**
    * @return     Returns the size of the given dimension. 0 being the first specified dimension.
    */
   size_t GetDimSize(size_t dimIndex) const
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
   template <size_t TDimIndex> size_t GetDimSize() const
   {
      return GetDimSizeImpl<TDimIndex, TDims...>();
   }

   /**
    * @return     Computes the flat index equivalent of the passed index pack.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   size_t ToFlatIndex(TIdx... indices) const
   {
      return ComputeFlatIndex<TDims...>(indices...);
   }

   /**
    * @return     The total number of elements in this matrix.
    */
   size_t GetFlatSize() const
   {
      return FlatSize;
   }

private:
   // Store the flat size of the array as a const in the base class.
   constexpr static size_t FlatSize = TotalVecSize<TDims...>();

   // Statically for this type stores the dimensions for runtime use.
   static constexpr const int mDimSizes[sizeof...(TDims)]{TDims...};

   // Available walkers.
   using WalkerNone = MatrixWalkerNoIndices<T, TDims...>;
   using WalkerFlat = MatrixWalkerFlatIndex<T, TDims...>;
   using WalkerInds = MatrixWalkerMatrixIndices<T, TDims...>;
};

/**
 * @brief      Writable interface to a matrix
 *
 * @tparam     T      value_type for the matrix
 * @tparam     TDims  A size_t list of dimensions
 */
template <template <typename, size_t...> class TAlloc, ScalarStateValue T, size_t... TDims>
   requires MatrixAlloc<TAlloc, T, TotalVecSize<TDims...>()>
class MatrixState : public virtual IMatrixState<T, TDims...>, public virtual ReadWriteState
{
public:
   using allocator_type = TAlloc<T, TDims...>;

   /**
    * @brief      Standard constructor.
    *
    * @param[in]  defaultScalar  The default value that initializes the matrix.
    */
   MatrixState(const T& defaultScalar) : mDefaultScalar(defaultScalar), mValue(mAllocator.GetVec())
   {
      SetToDefault();
   }

   /**
    * @return      The value at the given matrix index.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   T& operator()(TIdx... indices)
   {
      return mValue[this->ToFlatIndex(indices...)];
   }

   /**
    * @return     True if at default, False otherwise.
    */
   virtual bool IsAtDefault() const override
   {
      return this->All(
         [this](const T& value)
         {
            return value == mDefaultScalar;
         });
   }

   /**
    * @brief      Sets the current value to the default scalar for every element.
    */
   virtual void SetToDefault() override
   {
      SetValues(mDefaultScalar);
   }

   // Map

   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 0>
   MatrixState& Map(TFunc&& mapper)
   {
      MatrixMap<T, TDims...>::template Run<WalkerNone>(mValue, std::forward<TFunc>(mapper));
      SignalValueChange();
      return *this;
   }

   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 1, size_t>
   MatrixState& Map(TFunc&& mapper)
   {
      MatrixMap<T, TDims...>::template Run<WalkerFlat>(mValue, std::forward<TFunc>(mapper));
      SignalValueChange();
      return *this;
   }

   template <typename TFunc> MatrixState& Map(TFunc&& mapper)
   {
      MatrixMap<T, TDims...>::template Run<WalkerInds>(mValue, std::forward<TFunc>(mapper));
      SignalValueChange();
      return *this;
   }

   /**
    * @brief      Sets the value at a given flat index.
    *
    * @param[in]  value      The value
    * @param[in]  flatIndex  The flat index
    */
   virtual void SetValue(const T& value, size_t flatIndex)
   {
      assert(flatIndex < FlatSize);
      mValue[flatIndex] = value;
      SignalValueChange();
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
      setter(mValue, FlatSize);
      SignalValueChange();
      return *this;
   }

   /**
    * @brief      Returns a readonly view of the data.
    * @return     The readonly data view as a span.
    */
   virtual const std::span<const T, TotalVecSize<TDims...>()> GetValues() const override
   {
      return mValue;
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
      Map(
         [&value](T& currentValue)
         {
            currentValue = value;
         });

      return *this;
   }

private:
   constexpr static size_t FlatSize = TotalVecSize<TDims...>();
   TAlloc<T, TDims...> mAllocator;
   std::span<T, FlatSize> mValue;
   T mDefaultScalar;

   // Available walkers.
   using WalkerNone = MatrixWalkerNoIndices<T, TDims...>;
   using WalkerFlat = MatrixWalkerFlatIndex<T, TDims...>;
   using WalkerInds = MatrixWalkerMatrixIndices<T, TDims...>;
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
class VectorState : public MatrixState<TAlloc, T, TSize>, public virtual IVectorState<T, TSize>
{
public:
   /**
    * @brief      Standard constructor
    *
    * @param[in]  defaultScalar  The default value for every element in the vector state.
    */
   VectorState(const T& defaultScalar) : MatrixState<TAlloc, T, TSize>(defaultScalar)
   {
   }
};

/**
 * Helper definitions.
 */
template <ScalarStateValue T, size_t... TDims>
class StaticMatrixState : public MatrixState<MatrixStaticAlloc, T, TDims...>
{
public:
   StaticMatrixState(const T& defaultScalar)
      : MatrixState<MatrixStaticAlloc, T, TDims...>(defaultScalar)
   {
   }
};

template <ScalarStateValue T, size_t... TDims>
class HeapMatrixState : public MatrixState<MatrixHeapAlloc, T, TDims...>
{
public:
   HeapMatrixState(const T& defaultScalar)
      : MatrixState<MatrixHeapAlloc, T, TDims...>(defaultScalar)
   {
   }
};

template <ScalarStateValue T, size_t TSize>
class StaticVectorState : public VectorState<MatrixStaticAlloc, T, TSize>
{
public:
   StaticVectorState(const T& defaultScalar)
      : VectorState<MatrixStaticAlloc, T, TSize>(defaultScalar)
   {
   }
};

template <ScalarStateValue T, size_t TSize>
class HeapVectorState : public VectorState<MatrixHeapAlloc, T, TSize>
{
public:
   HeapVectorState(const T& defaultScalar) : VectorState<MatrixHeapAlloc, T, TSize>(defaultScalar)
   {
   }
};
}   // namespace kc