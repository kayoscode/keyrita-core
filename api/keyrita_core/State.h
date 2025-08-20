#pragma once

#include <concepts>
#include <cpp_events/Event.h>

namespace kc
{
/**
 * @brief      Defines the data packaged with a change of state.
 */
struct tStateChangedEventData : public tEventData
{
public:
   tStateChangedEventData() : tEventData()
   {
   }
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

   /**
    * @return     Compares if the current value is equal to default.
    */
   virtual bool IsAtDefault() = 0;

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
    * @return     True if desired value different, False otherwise.
    */
   virtual bool IsDesiredValueDifferent() = 0;

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
   ReadWriteState(){}

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

#pragma region Scalar state

template<typename T>
concept ScalarStateValue = std::copyable<T> && std::equality_comparable<T>;

template <ScalarStateValue T>
class IScalarState : public virtual ReadState
{
public:
   /**
    * @return     The queried scalar value.
    */
   virtual T GetValue() = 0;
};

template <ScalarStateValue T>
class ScalarState : public virtual IScalarState<T>, public virtual ReadWriteState
{
public:
   /**
    * @brief      Standard constructor
    *
    * @param[in]  defaultValue  Specify the default value.
    */
   ScalarState(T defaultValue)
      : mDefaultValue(defaultValue)
   {
      SetToDefault();
   }

   T GetValue() override 
   {
      return mValue;
   }

   /**
    * @brief      Sets the value to the given new value
    *
    * @param[in]  newValue  The new value to set to.
    */
   void Set(T newValue)
   {
      mDesiredValue = newValue;
      SetToDesiredValue();
   }

   /**
    * @brief      Sets the desired value to default, then attempts to change the state value.
    */
   void SetToDefault() override
   {
      Set(mDefaultValue);
   }

   /**
    * @return     Compares if the current value is equal to default.
    */
   bool IsAtDefault() override 
   {
      return this->mValue != mDefaultValue;
   }

protected:
   /**
    * @brief      Called after the state value changes.
    */
   virtual void OnChangeAction() override
   {
      tStateChangedEventData stateChangedEventData;

      // Push events to listeners.
      this->mOnChanged.Dispatch(stateChangedEventData);

      // Now call our generic child action.
      Action(stateChangedEventData);
   }

   /**
    * @brief      Simply set the value to the current desired value.
    */
   void ApplyDesiredValue() override
   {
      this->mValue = mDesiredValue;
   }

   /**
    * @return     True if desired value different, False otherwise.
    */
   bool IsDesiredValueDifferent() override
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
}   // namespace kc