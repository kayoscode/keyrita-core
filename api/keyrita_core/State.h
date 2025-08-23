#pragma once

#include <array>
#include <concepts>
#include <cpp_events/Event.h>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <utility>

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
   /**
    * @return     True if desired value different, False otherwise.
    */
   virtual bool IsDesiredValueDifferent() const = 0;
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
    * @brief      Initializes the value in the setting. Usually to some default.
    */
   virtual void Init() = 0;

   /**
    * @brief      Called after the state value changes.
    */
   virtual void OnChangeAction() = 0;

   /**
    * @brief      The state override determines what the desired value is. This method is called to
    * set the state to whatever that desired value is. This override should check if the desired
    * value is different from the current value. If it is, set the value and call OnChangeAction.
    */
   virtual void SetToDesiredValue() = 0;
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
   /**
    * @brief      Initializes the value in the setting. Usually to some default.
    */
   virtual void Init() override
   {
      SetToDefault();
   }

   /**
    * @brief      Checks if the desired value differs from the current value. If
    * so, mutate the state, and call the on change action.
    */
   virtual void SetToDesiredValue() override
   {
      if (IsDesiredValueDifferent())
      {
         ApplyDesiredValue();

         // This cannot fail, so continue on.
         OnChangeAction();
      }
   }

   /**
    * @brief      Called after the state value changes.
    */
   virtual void OnChangeAction() override
   {
      tStateChangedEventData stateChangedEventData(this);

      // Push events to listeners.
      this->mOnChanged.Dispatch(stateChangedEventData);

      // Now call our generic child action.
      Action(stateChangedEventData);
   }

   /**
    * @brief      Override logic to copy the desired value into the current value.
    */
   virtual void ApplyDesiredValue() = 0;

   /**
    * @brief      Override to set specific behavior for this setting after
    * the value has been changed and all registrants have been notified.
    */
   virtual void Action(const tStateChangedEventData& eventData)
   {
      // Nothing to do by default.
   }
};

template <typename T>
concept ScalarStateValue = std::copyable<T> && std::equality_comparable<T>;

#pragma region Scalar state

template <ScalarStateValue T> class IScalarState : public virtual IReadState
{
public:
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
      mDesiredValue = newValue;
      SetToDesiredValue();
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

protected:
   /**
    * @brief      Simply set the value to the current desired value.
    */
   virtual void ApplyDesiredValue() override
   {
      this->mValue = mDesiredValue;
   }

   /**
    * @return     True if desired value different, False otherwise.
    */
   virtual bool IsDesiredValueDifferent() const override
   {
      return this->mValue != mDesiredValue;
   }

private:
   /**
    * The current desired value.
    */
   T mDesiredValue;
   T mDefaultValue;
   T mValue;
};

#pragma endregion

#pragma region Vector state

template <ScalarStateValue T, size_t TSize> class IVectorState : public virtual IReadState
{
public:
   /**
    * @brief      Returns the value at a given index.
    *
    * @param[in]  index  The index
    *
    * @return     The value.
    */
   virtual T GetValue(size_t index) const = 0;

   /**
    * @brief       Returns the value at a given index.
    * @param       index
    * @return      the value at the given index
    */
   virtual T operator[](size_t index) const = 0;

   /**
    * @brief      Returns a readonly view of the data.
    *
    * @return     The readonly data view as a span.
    */
   virtual const std::span<const T> GetValue() const = 0;

   /**
    * @brief       Performs an action per element readonly
    * @param       func
    */
   virtual void ForEach(std::function<void(const T& value, size_t index)> func) const = 0;

   /**
    * @return      true if all values match the predicate
    */
   virtual bool All(std::function<bool(const T& value)> predicate) const = 0;

   /**
    * @return      true if any of the values in the set match the predicate
    */
   virtual bool Any(std::function<bool(const T& value)> predicate) const = 0;

   /**
    * @brief      Returns the number of elements in this vector state.
    * This cannot be modified at runtime.
    *
    * @return     The size.
    */
   size_t GetSize() const
   {
      return TSize;
   }
};

template <ScalarStateValue T, size_t TSize>
class VectorState : public virtual ReadWriteState, public virtual IVectorState<T, TSize>
{
public:
   /**
    * @brief      Standard constructor
    *
    * @param[in]  defaultScalar  The default value for every element in the vector state.
    */
   VectorState(const T& defaultScalar) : mDefaultScalar(defaultScalar)
   {
      // Need to have at least one element.
      static_assert(TSize > 0, "Invalid vector size");

      SetToDefault();
   }

   virtual T GetValue(size_t index) const override
   {
      assert(index < TSize);
      return this->mValue[index];
   }

   /**
    * @brief       Returns the value at the given index, no bounds checking.
    * @param       index
    * @return
    */
   virtual T operator[](size_t index) const override
   {
      return mValue[index];
   }

   virtual const std::span<const T> GetValue() const override
   {
      return this->mValue;
   }

   /**
    * @brief      Sets a single value and updates the state.
    *
    * @param[in]  value  The value
    * @param[in]  index  The index
    */
   virtual void SetValue(const T& value, size_t index)
   {
      assert(index < TSize);

      mDesiredValue[index] = value;
      SetToDesiredValue();
   }

   /**
    * @brief       Sets all the values in the vector to the given value.
    * @param       value
    */
   virtual VectorState& SetAll(const T& value)
   {
      SetValues(
         [value](std::span<T> data, size_t count)
         {
            for (size_t i = 0; i < count; i++)
            {
               data[i] = value;
            }
         });

      return *this;
   }

   /**
    * @brief      Sets many values and only updates the state and calls the changed callback once.
    *
    * @param[in]  setCallback  The set callback
    */
   virtual VectorState& SetValues(std::function<void(std::span<T> data, size_t count)> setCallback)
   {
      setCallback(mDesiredValue, TSize);
      SetToDesiredValue();
      return *this;
   }

   /**
    * @return     Compares if the current value is equal to default.
    */
   virtual bool IsAtDefault() const override
   {
      for (size_t i = 0; i < TSize; i++)
      {
         if (mValue[i] != mDefaultScalar)
         {
            return false;
         }
      }

      return true;
   }

   /**
    * @brief       Sets every value to the default scalar
    */
   virtual void SetToDefault() override
   {
      SetAll(mDefaultScalar);
   }

   virtual bool All(std::function<bool(const T& value)> predicate) const override
   {
      for (int i = 0; i < TSize; i++)
      {
         if (!predicate(mValue[i]))
         {
            return false;
         }
      }

      return true;
   }

   virtual bool Any(std::function<bool(const T& value)> predicate) const override
   {
      for (int i = 0; i < TSize; i++)
      {
         if (predicate(mValue[i]))
         {
            return true;
         }
      }

      return false;
   }

   /**
    * @brief       Transforms each value to a new value from the mapper func
    * @param       mapper
    */
   virtual VectorState& Map(std::function<T(const T& value)> mapper)
   {
      for (int i = 0; i < TSize; i++)
      {
         mDesiredValue[i] = mapper(mValue[i]);
      }

      SetToDesiredValue();
      return *this;
   }

   /**
    * @brief       Performs a given action for each element in the vector
    * @param       The action to perform for each element.
    */
   virtual void ForEach(std::function<void(const T& value, size_t index)> action) const override
   {
      for (int i = 0; i < TSize; i++)
      {
         action(mValue[i], i);
      }
   }

   /**
    * @brief       Generates list values for each element based on the tabulation func.
    * @param       func
    */
   virtual VectorState& Tabulate(std::function<T(const T& oldValue, size_t index)> func)
   {
      for (size_t i = 0; i < TSize; i++)
      {
         mDesiredValue[i] = func(mValue[i], i);
      }

      SetToDesiredValue();
      return *this;
   }

protected:
   /**
    * @brief      Simply set the value to the current desired value.
    */
   virtual void ApplyDesiredValue() override
   {
      for (size_t i = 0; i < TSize; i++)
      {
         mValue[i] = mDesiredValue[i];
      }
   }

   /**
    * @return     True if desired value different, False otherwise.
    */
   virtual bool IsDesiredValueDifferent() const override
   {
      for (size_t i = 0; i < TSize; i++)
      {
         if (mValue[i] != mDesiredValue[i])
         {
            return true;
         }
      }

      return false;
   }

private:
   T mValue[TSize];
   T mDesiredValue[TSize];
   T mDefaultScalar;
};

#pragma endregion

#pragma region Matrix state

/**
 * @return      Returns the number of dimensions given by a templated list of dims
 */
template <size_t... TDims> constexpr int GetNumDims()
{
   return sizeof...(TDims);
}

template <size_t TDim> constexpr size_t TotalVecSize()
{
   return TDim;
}

template <size_t TFirstDim, size_t... TRemainingDims>
   requires(sizeof...(TRemainingDims) > 0)
constexpr size_t TotalVecSize()
{
   return TFirstDim * TotalVecSize<TRemainingDims...>();
}

/**
 * Concept for indexing the matrix
 */
template <size_t TNumDims, typename... TIdx>
concept MatrixIndices = (sizeof...(TIdx) == TNumDims) && (std::convertible_to<TIdx, size_t> && ...);

/**
 * @brief      Base case to compute flat index.
 */
template <size_t TDim, typename TIdx> constexpr size_t ComputeFlatIndexRecursive(TIdx idx)
{
   return idx;
}

/**
 * @brief      Recursive case for computing a flat index.
 */
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

// Concept for an action done for each element in a matrix.
template <typename T, typename TFunc>
concept MatrixAction = ScalarStateValue<T> && requires(TFunc predicate, const T& value) {
   { predicate(value) } -> std::same_as<void>;
};

// Concept for a predicate for each element in a matrix.
template <typename T, typename TFunc>
concept MatrixPredicate = ScalarStateValue<T> && requires(TFunc predicate, const T& value) {
   { predicate(value) } -> std::convertible_to<bool>;
};

// Concept for a mapping operation for a matrix
template <typename T, typename TFunc>
concept MatrixMap = ScalarStateValue<T> && requires(TFunc predicate, const T& value) {
   { predicate(value) } -> std::convertible_to<T>;
};

#pragma region Matrix Walkers

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
      { func(value, indices...) } -> std::same_as<void>;
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
      { func(value, indices...) } -> std::same_as<void>;
   };

/**
 * Matrix walker that iterates through each element in the matrix not maintaing any indices.
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
         action(value);
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
         action(matrixValues[i], i);
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
   template <typename TSpanType, typename TFunc, size_t TCurrentDimIdx, size_t TFirstDim,
      size_t... TRemainingDims>
   static void WalkImpl(std::span<TSpanType, TotalVecSize<TDims...>()> matrixValues, TFunc&& func,
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
            WalkImpl<TSpanType, TFunc, TCurrentDimIdx + 1, TRemainingDims...>(
               matrixValues, std::forward<TFunc>(func), indices, flatIdx);
         }
         else
         {
            std::apply(
               [&](auto... idx)
               {
                  func(matrixValues[flatIdx], idx...);
               },
               indices);
         }

         // Increase the offset by the stride to move to the next dimension.
         flatIdx += flatIdxStride;
      }
   }
};

#pragma endregion

/**
 * @brief      An abstraction on top of vector state providing utilities to handle N dimensions
 */
template <ScalarStateValue T, size_t... TDims> class IMatrixState : public virtual IReadState
{
public:
   /**
    * @brief      Returns a readonly view of the data.
    *
    * @return     The readonly data view as a span.
    */
   virtual const std::span<const T, TotalVecSize<TDims...>()> GetValue() const = 0;

   /**
    * @brief       Performs an action per element readonly.
    * @param       The action to perform per element
    */
   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 0>
   void ForEach(TFunc&& action) const
   {
      // Walk through the array with no indices and perform the action.
      ForEachImpl<MatrixWalkerNoIndices<T, TDims...>>(GetValue(), std::forward<TFunc>(action));
   }

   /**
    * Performs foreach operation passing the flat index during iteration.
    */
   template <typename TFunc>
      requires MatrixImmutableWalkClient<T, TFunc, 1, size_t>
   void ForEach(TFunc&& action) const
   {
      // Walk through the array with no indices and perform the action.
      IMatrixState<T, TDims...>::ForEachImpl<MatrixWalkerFlatIndex<T, TDims...>>(
         GetValue(), std::forward<TFunc>(action));
   }

   /**
    * Performs a foreach operation passing the matrix raw indices during iteration.
    */
   template <typename TFunc> void ForEach(TFunc&& action) const
   {
      IMatrixState<T, TDims...>::ForEachImpl<MatrixWalkerMatrixIndices<T, TDims...>>(
         GetValue(), std::forward<TFunc>(action));
   }

   /**
    * @brief      Counts the number of elements in the matrix that match the predicate
    *
    * @param      predicate  The predicate
    *
    * @return     The total number of elements matching the predicate.
    */
   template <typename TFunc>
      requires MatrixPredicate<T, TFunc>
   size_t CountIf(TFunc&& predicate) const
   {
      size_t count = 0;
      MatrixWalkerNoIndices<T, FlatSize>::WalkReadOnly(GetValue(),
         [&count, predicate = std::forward<TFunc>(predicate)](const T& value)
         {
            if (predicate(value))
            {
               count++;
            }
         });

      return count;
   }

   /**
    * @return      true if all values match the predicate
    */
   template <typename TFunc>
      requires MatrixPredicate<T, TFunc>
   bool All(TFunc&& predicate) const
   {
      std::span<const T, FlatSize> valueSpan = GetValue();

      for (const T& v : valueSpan)
      {
         if (!predicate(v))
         {
            return false;
         }
      }

      return true;
   }

   /**
    * @return      true if any of the values in the set match the predicate
    */
   template <typename TFunc>
      requires MatrixPredicate<T, TFunc>
   bool Any(TFunc&& predicate) const
   {
      std::span<const T, FlatSize> valueSpan = GetValue();

      for (const T& v : valueSpan)
      {
         if (predicate(v))
         {
            return true;
         }
      }

      return false;
   }

   /**
    * @brief      Returns the value of the matrix at the given index pack
    *
    * @param[in]  indices  The indices
    *
    * @tparam     TIdx     The indices to get the value at.
    *
    * @return     The matrix value at the given index.
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   T GetMatrixValue(TIdx... indices) const
   {
      return this->GetValue(ToFlatIndex(indices...));
   }

   /**
    * @return      Returns the value of the matrix at the given flat index.
    */
   virtual T GetValue(size_t flatIndex) const = 0;

   /**
    * @brief      Returns the total number of elements that appear in the matrix.
    * @return     The size.
    */
   size_t GetSize() const
   {
      return TotalVecSize<TDims...>();
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
   template <typename TWalker, typename TFunc>
   void ForEachImpl(std::span<const T, TotalVecSize<TDims...>()> matrixValues, TFunc&& f) const
   {
      TWalker::WalkReadOnly(matrixValues, std::forward<TFunc>(f));
   }

   // Store the flat size of the array as a const in the base class.
   constexpr static size_t FlatSize = TotalVecSize<TDims...>();

   // Statically for this type stores the dimensions for runtime use.
   static constexpr const int mDimSizes[sizeof...(TDims)]{TDims...};
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
   MatrixState(const T& defaultScalar)
      : mDefaultScalar(defaultScalar), mValue(nullptr), mDesiredValue(nullptr)
   {
      mValue = std::make_unique<std::array<T, FlatSize>>();
      mDesiredValue = std::make_unique<std::array<T, FlatSize>>();
      SetToDefault();
   }

   /**
    * @brief      Returns a readonly view of the data.
    *
    * @return     The readonly data view as a span.
    */
   virtual const std::span<const T, TotalVecSize<TDims...>()> GetValue() const override
   {
      return *mValue;
   }

   /**
    * @return     True if at default, False otherwise.
    */
   virtual bool IsAtDefault() const override
   {
      for (size_t i = 0; i < FlatSize; i++)
      {
         if (mValue->at(i) != mDefaultScalar)
         {
            return false;
         }
      }

      return true;
   }

   /**
    * @brief      Sets the current value to the default scalar for every element.
    */
   virtual void SetToDefault() override
   {
      for (size_t i = 0; i < FlatSize; i++)
      {
         mDesiredValue->at(i) = mDefaultScalar;
      }

      SetToDesiredValue();
   }

   /**
    * @return     The value at a given flat index.
    */
   virtual T GetValue(size_t flatIndex) const override
   {
      return mValue->at(flatIndex);
   }

   /**
    * @brief      Sets the value at a given flat index.
    *
    * @param[in]  value      The value
    * @param[in]  flatIndex  The flat index
    */
   virtual void SetValue(const T& value, size_t flatIndex)
   {
      assert(flatIndex < this->FlatSize);

      mDesiredValue->at(flatIndex) = value;
      SetToDesiredValue();
   }

   /**
    * Sets a valid in the matrix
    */
   template <typename... TIdx>
      requires MatrixIndices<sizeof...(TDims), TIdx...>
   MatrixState<T, TDims...> SetMatrixValue(T value, TIdx... indices)
   {
      this->SetValue(value, this->ToFlatIndex(indices...));
      return *this;
   }

   /**
    * Maps every element in the matrix to a new value given only the current value.
    */
   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 0>
   MatrixState<T, TDims...>& Map(TFunc&& mapper)
   {
      MatrixWalkerNoIndices<T, TDims...>::WalkReadWrite(
         *mDesiredValue, std::forward<TFunc>(mapper));
      SetToDesiredValue();
      return *this;
   }

   /**
    * Maps every element in the matrix to a new value given the flat index.
    */
   template <typename TFunc>
      requires MatrixMutableWalkClient<T, TFunc, 1, size_t>
   MatrixState<T, TDims...>& Map(TFunc&& mapper)
   {
      MatrixWalkerFlatIndex<T, TDims...>::WalkReadWrite(
         *mDesiredValue, std::forward<TFunc>(mapper));
      SetToDesiredValue();
      return *this;
   }

   /**
    * Maps every element in the matrix to a new value given the entire matrix index.
    */
   template <typename TFunc> MatrixState<T, TDims...>& Map(TFunc&& mapper)
   {
      MatrixWalkerMatrixIndices<T, TDims...>::WalkReadWrite(
         *mDesiredValue, std::forward<TFunc>(mapper));
      SetToDesiredValue();
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
      setter(*mDesiredValue, FlatSize);
      SetToDesiredValue();
      return *this;
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
      for (size_t i = 0; i < FlatSize; i++)
      {
         mDesiredValue[i] = value;
      }

      SetToDesiredValue();
      return *this;
   }

private:
   virtual bool IsDesiredValueDifferent() const override
   {
      return true;
   }

   virtual void ApplyDesiredValue() override
   {
      // Swap the pointers for more speed.
      std::swap(mValue, mDesiredValue);
   }

   constexpr static size_t FlatSize = TotalVecSize<TDims...>();
   std::unique_ptr<std::array<T, FlatSize>> mValue;
   std::unique_ptr<std::array<T, FlatSize>> mDesiredValue;
   T mDefaultScalar;
};
}   // namespace kc