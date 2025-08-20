#include "keyrita_core/State.h"
#include "gtest/gtest.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace kc;

/**
 * We want to test that state can be retrieved and set based on
 * a given interface.
 */
TEST(StateTests, DoubleStateTests)
{
   double def = 100;
   ScalarState<double> testScalar(def);
   IReadState& testReadScalar = testScalar;

   ASSERT_TRUE(testScalar.IsAtDefault());

   int writeCounter = 0;

   testReadScalar.OnChanged().Register([&writeCounter, &testScalar](const tStateChangedEventData& data) 
   {
      ASSERT_EQ(data.SourceState, &testScalar);
      writeCounter++;
   });

   // Start by trying to write default.
   testScalar.Set(def);

   // This should not trigger the value to change, nor hit the
   // change notification. 
   ASSERT_EQ(testScalar.GetValue(), def);
   ASSERT_EQ(writeCounter, 0);
   ASSERT_TRUE(testScalar.IsAtDefault());

   // After setting it to zero, we should see the value change, and the change notification get hit.
   testScalar.Set(0);
   ASSERT_EQ(writeCounter, 1);
   ASSERT_EQ(testScalar.GetValue(), 0);
   ASSERT_FALSE(testScalar.IsAtDefault());

   testScalar.Set(0.0001);
   ASSERT_EQ(writeCounter, 2);
   ASSERT_EQ(testScalar.GetValue(), 0.0001);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Reset back to default.
   testScalar.SetToDefault();
   ASSERT_EQ(writeCounter, 3);
   ASSERT_EQ(testScalar.GetValue(), def);
   ASSERT_TRUE(testScalar.IsAtDefault());

   // Nan doesn't check for equality, so both of these writes should trigger the change notification.
   testScalar.Set(NAN);
   testScalar.Set(NAN);
   ASSERT_EQ(writeCounter, 5);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Infinty does check for equality, so this should only trigger once.
   testScalar.Set(INFINITY);
   testScalar.Set(INFINITY);
   ASSERT_EQ(writeCounter, 6);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Infinty does check for equality, so this should only trigger once.
   testScalar.Set(-INFINITY);
   testScalar.Set(-INFINITY);
   ASSERT_EQ(writeCounter, 7);
   ASSERT_FALSE(testScalar.IsAtDefault());
}

enum class eTestEnum
{
   Item1,
   Item2,
   Item3,
   ItemCount,
};

/**
 * I'd like to verify that state works correctly with enumerations.
 */
TEST(StateTests, EnumStateTests)
{
   eTestEnum def = eTestEnum::Item2;
   ScalarState<eTestEnum> enumState(def);
   ASSERT_TRUE(enumState.IsAtDefault());

   int writeCounter = 0;

   enumState.OnChanged().Register([&writeCounter, &enumState](const tStateChangedEventData& data) 
   {
      ASSERT_EQ(data.SourceState, &enumState);
      writeCounter++;
   });

   // Start by trying to write default.
   enumState.Set(def);

   // This should not trigger the value to change, nor hit the
   // change notification. 
   ASSERT_EQ(enumState.GetValue(), def);
   ASSERT_EQ(writeCounter, 0);
   ASSERT_TRUE(enumState.IsAtDefault());

   // After setting it to zero, we should see the value change, and the change notification get hit.
   enumState.Set(eTestEnum::Item1);
   ASSERT_EQ(writeCounter, 1);
   ASSERT_EQ(enumState.GetValue(), eTestEnum::Item1);
   ASSERT_FALSE(enumState.IsAtDefault());

   enumState.Set(eTestEnum::Item3);
   ASSERT_EQ(writeCounter, 2);
   ASSERT_EQ(enumState.GetValue(), eTestEnum::Item3);
   ASSERT_FALSE(enumState.IsAtDefault());

   // Reset back to default.
   enumState.SetToDefault();
   ASSERT_EQ(writeCounter, 3);
   ASSERT_EQ(enumState.GetValue(), def);
   ASSERT_TRUE(enumState.IsAtDefault());
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
