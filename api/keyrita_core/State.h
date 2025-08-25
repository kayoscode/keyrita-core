#pragma once

#include "MatrixQuery.h"

#include <array>
#include <cpp_events/Event.h>
#include <cstddef>
#include <cstring>
#include <span>
#include <utility>
#include <wchar.h>

namespace kc
{
class IReadState;

/**
 * @brief      Defines the data packaged with a change of state.
 */
struct tStateChangedEventData : public tEventData
{
public:
   /**
    * Default constructor.
    */
   tStateChangedEventData() : tEventData(), SourceState(nullptr)
   {
   }

   /**
    * Standard constructor.
    */
   tStateChangedEventData(IReadState* sourceState) : tEventData(), SourceState(sourceState)
   {
   }

   // The state value from which this event originated.
   IReadState* SourceState;
};

/**
 * @brief      Interface for state that can only be read from.
 */
class IReadState
{
public:
   /**
    * @brief      Allows clients to register to changed events for this state object.
    * @return     A reference to the state changed event listener.
    */
   virtual EventListener<tStateChangedEventData>& OnChanged() = 0;

   /**
    * @return     Compares if the current value is equal to default.
    */
   virtual bool IsAtDefault() const = 0;

protected:
};

/**
 * @brief      Defines the contract that must be implemented for any read write state
 */
class IReadWriteState : public virtual IReadState
{
public:
   /**
    * @brief      Sets the desired value to default, then attempts to change the state value.
    */
   virtual void SetToDefault() = 0;

protected:
   /**
    * @brief      Called after the state value changes.
    */
   virtual void OnChangeAction() = 0;
};

class ReadState : public virtual IReadState
{
public:
   /**
    * @return     A listener to the on changed event.
    */
   EventListener<tStateChangedEventData>& OnChanged() override
   {
      return mOnChanged;
   }

protected:
   /**
    * The event dispatcher for when the state has changed.
    */
   StandardEventDispatcher<tStateChangedEventData> mOnChanged;
};

class ReadWriteState : public virtual ReadState, public virtual IReadWriteState
{
public:
   /**
    * @brief      Standard constructor, initializes to a default.
    */
   ReadWriteState()
   {
   }

protected:
   void SignalValueChange()
   {
      OnChangeAction();
      // Now call our generic child action.

      Action();
   }

   /**
    * @brief      Override to set specific behavior for this setting after
    * the value has been changed and all registrants have been notified.
    */
   virtual void Action()
   {
      // Nothing to do by default.
   }

private:
   /**
    * @brief      Called after the state value changes.
    */
   virtual void OnChangeAction() override
   {
      tStateChangedEventData stateChangedEventData(this);

      // Push events to listeners.
      this->mOnChanged.Dispatch(stateChangedEventData);
   }
};

template <ScalarStateValue T> class IScalarState : public virtual IReadState
{
public:
   using value_type = T;

   /**
    * @return     The queried scalar value.
    */
   virtual const T& GetValue() = 0;
};

/**
 * @brief       Stores and handles scalar state.
 * When it comes to floating point, it's important to remember that normal floating point
 * comparisons apply. NAN equal to anything is false (even itself). If you set it to NaN, Even
 * setting it back to nan will trigger a state change notification. Ideally, you wouldn't allow Nan
 * to be a valid value for this kind of state. If you'd like extra logic to correct for this, make a
 * derivation on top of this class and implement the logic in the equality comparisons.
 */
template <ScalarStateValue T>
class ScalarState : public virtual ReadWriteState, public virtual IScalarState<T>
{
public:
   /**
    * @brief      Standard constructor
    *
    * @param[in]  defaultValue  Specify the default value.
    */
   ScalarState(const T& defaultValue) : mDefaultValue(defaultValue)
   {
      SetToDefault();
   }

   const T& GetValue() override
   {
      return mValue;
   }

   /**
    * @brief      Sets the value to the given new value
    *
    * @param[in]  newValue  The new value to set to.
    */
   void Set(const T& newValue)
   {
      mValue = newValue;
      SignalValueChange();
   }

   /**
    * @brief      Sets the desired value to default, then attempts to change the state value.
    */
   virtual void SetToDefault() override
   {
      Set(mDefaultValue);
   }

   /**
    * @return     Compares if the current value is equal to default.
    */
   virtual bool IsAtDefault() const override
   {
      return this->mValue == mDefaultValue;
   }

private:
   /**
    * The current desired value.
    */
   T mDefaultValue;
   T mValue;
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

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 0>
   void ForEach(TFunc&& f) const
   {
      MatrixForEach<T, TDims...>::template Impl<WalkerNone>(GetValues(), std::forward<TFunc>(f));
   }

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 1, size_t>
   void ForEach(TFunc&& f) const
   {
      MatrixForEach<T, TDims...>::template Impl<WalkerFlat>(GetValues(), std::forward<TFunc>(f));
   }

   template <typename TFunc> void ForEach(TFunc&& f) const
   {
      MatrixForEach<T, TDims...>::template Impl<WalkerInds>(GetValues(), std::forward<TFunc>(f));
   }

   // CountIf

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 0>
   size_t CountIf(TFunc&& f) const
   {
      return MatrixCountIf<T, TDims...>::template Impl<WalkerNone>(
         GetValues(), std::forward<TFunc>(f));
   }

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 1, size_t>
   size_t CountIf(TFunc&& f) const
   {
      return MatrixCountIf<T, TDims...>::template Impl<WalkerFlat>(
         GetValues(), std::forward<TFunc>(f));
   }

   template <typename TFunc> size_t CountIf(TFunc&& f) const
   {
      return MatrixCountIf<T, TDims...>::template Impl<WalkerInds>(
         GetValues(), std::forward<TFunc>(f));
   }

   // All

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 0>
   bool All(TFunc&& predicate) const
   {
      return MatrixAllQuery<T, TDims...>::template Impl<WalkerNone>(
         GetValues(), std::forward<TFunc>(predicate));
   }

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 1, size_t>
   bool All(TFunc&& predicate) const
   {
      return MatrixAllQuery<T, TDims...>::template Impl<WalkerFlat>(
         GetValues(), std::forward<TFunc>(predicate));
   }

   template <typename TFunc> bool All(TFunc&& predicate) const
   {
      return MatrixAllQuery<T, TDims...>::template Impl<WalkerInds>(
         GetValues(), std::forward<TFunc>(predicate));
   }

   // Any
   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 0>
   bool Any(TFunc&& predicate) const
   {
      return MatrixAnyQuery<T, TDims...>::template Impl<WalkerNone>(
         GetValues(), std::forward<TFunc>(predicate));
   }

   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 1, size_t>
   bool Any(TFunc&& predicate) const
   {
      return MatrixAnyQuery<T, TDims...>::template Impl<WalkerFlat>(
         GetValues(), std::forward<TFunc>(predicate));
   }

   template <typename TFunc> bool Any(TFunc&& predicate) const
   {
      return MatrixAnyQuery<T, TDims...>::template Impl<WalkerInds>(
         GetValues(), std::forward<TFunc>(predicate));
   }

   // Fold

   template <typename TFoldResult = T, typename TFunc>
      requires MatrixFoldClient<TFoldResult, T, TFunc, 0>
   void Fold(TFoldResult& initialValue, TFunc&& func) const
   {
      MatrixFoldQuery<T, TDims...>::template Impl<WalkerNone>(
         initialValue, GetValues(), std::forward<TFunc>(func));
   }

   template <typename TFoldResult = T, typename TFunc>
      requires MatrixFoldClient<TFoldResult, T, TFunc, 1, size_t>
   void Fold(TFoldResult& initialValue, TFunc&& func) const
   {
      MatrixFoldQuery<T, TDims...>::template Impl<WalkerFlat>(
         initialValue, GetValues(), std::forward<TFunc>(func));
   }

   template <typename TFoldResult = T, typename TFunc>
   void Fold(TFoldResult& initialValue, TFunc&& func) const
   {
      MatrixFoldQuery<T, TDims...>::template Impl<WalkerInds>(
         initialValue, GetValues(), std::forward<TFunc>(func));
   }

   // FindIf

   template <typename TFunc, typename... TIdx>
      requires(sizeof...(TIdx) == sizeof...(TDims) || sizeof...(TIdx) == 1)
   bool FindIf(TFunc&& predicate, TIdx&... indices) const
   {
      return MatrixFindIfQuery<T, TDims...>::Impl(
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

template <ScalarStateValue T, size_t... TDims>
class MatrixState : public virtual IMatrixState<T, TDims...>, public virtual ReadWriteState
{
public:
   /**
    * @brief      Standard constructor.
    *
    * @param[in]  defaultScalar  The default value that initializes the matrix.
    */
   MatrixState(const T& defaultScalar) : mDefaultScalar(defaultScalar), mValue(nullptr)
   {
      mValue = std::make_unique<std::array<T, FlatSize>>();
      SetToDefault();
   }

   /**
    * @return      The value at the given matrix index.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   T& operator()(TIdx... indices)
   {
      return mValue->at(this->ToFlatIndex(indices...));
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
   MatrixState<T, TDims...>& Map(TFunc&& mapper)
   {
      MatrixMap<T, TDims...>::template Impl<WalkerNone>(*mValue, std::forward<TFunc>(mapper));
      SignalValueChange();
      return *this;
   }

   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 1, size_t>
   MatrixState<T, TDims...>& Map(TFunc&& mapper)
   {
      MatrixMap<T, TDims...>::template Impl<WalkerFlat>(*mValue, std::forward<TFunc>(mapper));
      SignalValueChange();
      return *this;
   }

   template <typename TFunc> MatrixState<T, TDims...>& Map(TFunc&& mapper)
   {
      MatrixMap<T, TDims...>::template Impl<WalkerInds>(*mValue, std::forward<TFunc>(mapper));
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
      mValue->at(flatIndex) = value;
      SignalValueChange();
   }

   /**
    * @brief      Sets a value at a given matrix pack.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   MatrixState<T, TDims...>& SetValue(T value, TIdx... indices)
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
   MatrixState<T, TDims...>& SetValues(TFunc&& setter)
   {
      setter(*mValue, FlatSize);
      SignalValueChange();
      return *this;
   }

   /**
    * @brief      Returns a readonly view of the data.
    * @return     The readonly data view as a span.
    */
   virtual const std::span<const T, TotalVecSize<TDims...>()> GetValues() const override
   {
      return *mValue;
   }

   /**
    * @brief      Sets every value in the matrix to the value
    *
    * @param[in]  value  The value
    *
    * @return     Self
    */
   MatrixState<T, TDims...>& SetValues(const T& value)
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
   std::unique_ptr<std::array<T, FlatSize>> mValue;
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
template <ScalarStateValue T, size_t TSize>
class VectorState : public virtual MatrixState<T, TSize>, public virtual IVectorState<T, TSize>
{
public:
   /**
    * @brief      Standard constructor
    *
    * @param[in]  defaultScalar  The default value for every element in the vector state.
    */
   VectorState(const T& defaultScalar) : MatrixState<T, TSize>(defaultScalar)
   {
   }
};
}   // namespace kc