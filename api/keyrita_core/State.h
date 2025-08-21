#pragma once

#include <concepts>
#include <cpp_events/Event.h>
#include <functional>
#include <span>

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

template <ScalarStateValue T> class IScalarState : public virtual ReadState
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

template <ScalarStateValue T, size_t TSize> class IVectorState : public virtual ReadState
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
   VectorState(const T& defaultScalar)
      : mDefaultScalar(defaultScalar)
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
      SetValues([value](std::span<T> data, size_t count) 
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
   virtual VectorState& Map(std::function<T (const T& value)> mapper)
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
}   // namespace kc