#include "keyrita_core/State.h"
#include <iostream>

using namespace kc;

int main()
{
   ScalarState<double> rwState(10);
   IReadState& roState = rwState;

   rwState.OnChanged().Register([] (const tStateChangedEventData& data)
   {
      std::cout << "value changed!\n";
   });

   // Same thing, so we should get 2 notifications
   roState.OnChanged().Register([] (const tStateChangedEventData& data)
   {
      std::cout << "value changed!\n";
   });

   std::cout << rwState.GetValue() << "\n";

   rwState.Set(100);
   rwState.Set(101);
   rwState.Set(101);

   std::cout << rwState.GetValue() << "\n";
   return 0;
}