#pragma once

#include "keyrita_core/State.h"

namespace kc
{
/**
 * @brief      Readonly interface to a scalar state value
 *
 * @tparam     T     value_type for the scalar
 */
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
}