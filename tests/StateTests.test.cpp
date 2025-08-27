#include "keyrita_core/MatrixQuery.h"
#include "keyrita_core/MatrixState.h"
#include "keyrita_core/MatrixUtils.h"
#include "keyrita_core/ScalarState.h"
#include "keyrita_core/State.h"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace kc;

/**
 * @param[in]  n     The triangle number to compute.
 * @return     The Nth triangle number
 */
constexpr int Trianglate(int n)
{
   return n * (n + 1) / 2;
}

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

   testReadScalar.OnChanged().Register(
      [&writeCounter, &testScalar](const tStateChangedEventData& data)
      {
         ASSERT_EQ(data.SourceState, &testScalar);
         writeCounter++;
      });

   // Start by trying to write default.
   testScalar.Set(def);

   // This should not trigger the value to change, nor hit the
   // change notification.
   ASSERT_EQ(testScalar.GetValue(), def);
   ASSERT_TRUE(testScalar.IsAtDefault());

   // After setting it to zero, we should see the value change, and the change notification get hit.
   testScalar.Set(0);
   ASSERT_EQ(testScalar.GetValue(), 0);
   ASSERT_FALSE(testScalar.IsAtDefault());

   testScalar.Set(0.0001);
   ASSERT_EQ(testScalar.GetValue(), 0.0001);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Reset back to default.
   testScalar.SetToDefault();
   ASSERT_EQ(testScalar.GetValue(), def);
   ASSERT_TRUE(testScalar.IsAtDefault());

   // Nan doesn't check for equality, so both of these writes should trigger the change
   // notification.
   testScalar.Set(NAN);
   testScalar.Set(NAN);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Infinty does check for equality, so this should only trigger once.
   testScalar.Set(INFINITY);
   testScalar.Set(INFINITY);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Infinty does check for equality, so this should only trigger once.
   testScalar.Set(-INFINITY);
   testScalar.Set(-INFINITY);
   ASSERT_FALSE(testScalar.IsAtDefault());

   // Make sure the callback got called.
   ASSERT_TRUE(writeCounter > 0);
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

   enumState.OnChanged().Register(
      [&writeCounter, &enumState](const tStateChangedEventData& data)
      {
         ASSERT_EQ(data.SourceState, &enumState);
         writeCounter++;
      });

   // Start by trying to write default.
   enumState.Set(def);

   // This should not trigger the value to change, nor hit the
   // change notification.
   ASSERT_EQ(enumState.GetValue(), def);
   ASSERT_TRUE(enumState.IsAtDefault());

   // After setting it to zero, we should see the value change, and the change notification get hit.
   enumState.Set(eTestEnum::Item1);
   ASSERT_EQ(enumState.GetValue(), eTestEnum::Item1);
   ASSERT_FALSE(enumState.IsAtDefault());

   enumState.Set(eTestEnum::Item3);
   ASSERT_EQ(enumState.GetValue(), eTestEnum::Item3);
   ASSERT_FALSE(enumState.IsAtDefault());

   // Reset back to default.
   enumState.SetToDefault();
   ASSERT_EQ(enumState.GetValue(), def);
   ASSERT_TRUE(enumState.IsAtDefault());

   // Make sure the callback got called.
   ASSERT_TRUE(writeCounter > 0);
}

TEST(StateTests, GeneralVectorStateTests)
{
   StaticVectorState<int, 30> vec(1);
   IVectorState<int, 30>& roVec = vec;

   // Check if default works.
   ASSERT_TRUE(roVec.All(
      [](int value)
      {
         return value == 1;
      }));

   int writeCount = 0;
   roVec.OnChanged().Register(
      [&writeCount](const tStateChangedEventData& data)
      {
         writeCount++;
      });

   ASSERT_EQ(vec.GetFlatSize(), roVec.GetFlatSize());
   ASSERT_EQ(vec.GetFlatSize(), 30);

   // Test the foreach interface as well as the values.
   roVec.ForEach(
      [&vec](const int& value, size_t index)
      {
         // Every value should be one and match the read write state.
         ASSERT_EQ(value, 1);
         ASSERT_EQ(value, vec[index]);
         ASSERT_EQ(value, vec.GetValue(index));
      });

   // Test the writable interface. The callback shouldn't be triggered since the value still hasn't
   // been changed from the default.
   vec.SetValue(1, 10);
   vec.SetValue(2, 0);
   ASSERT_EQ(roVec[0], 2);

   // Every other value should still be default.
   for (size_t i = 1; i < 30; i++)
   {
      ASSERT_EQ(vec[i], 1);
   }

   ASSERT_TRUE(roVec.Any(
      [](int value)
      {
         return value == 1;
      }));

   ASSERT_FALSE(vec.IsAtDefault());
   vec.SetToDefault();
   ASSERT_TRUE(vec.IsAtDefault());

   vec.SetValues(2);

   ASSERT_FALSE(vec.IsAtDefault());

   // Check a few of the functional things
   vec.Map(
         [](int& value, size_t index)
         {
            value = index + 1;
         })
      .Map(
         [](int& value)
         {
            value *= 2;
         })
      .ForEach(
         [](int value, size_t index)
         {
            ASSERT_EQ(value, (index + 1) * 2);
         });

   // Finally test set values
   vec.SetToDefault();
   ASSERT_TRUE(roVec.IsAtDefault());

   vec.SetValues(
      [](std::span<int> values, size_t count)
      {
         for (size_t i = 0; i < count; i++)
         {
            values[i] = count;
         }
      });

   ASSERT_TRUE(roVec.All(
      [](int value)
      {
         return value == 30;
      }));

   // Check that the callback has been called.
   ASSERT_TRUE(writeCount > 0);
}

typedef size_t func_test_t;

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixQueries
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   constexpr static void Test()
   {
      // Set to something that's non zero.
      mat_t matrix(10);

      // Test defaulting.
      MatrixUtils::FillSequence(matrix);
      ASSERT_FALSE(matrix.IsAtDefault());
      matrix.SetToDefault();
      ASSERT_TRUE(matrix.IsAtDefault());

      // Test size and dimensions query.
      size_t expectedSize = 1;
      for (size_t i = 0; i < matrix.GetNumDims(); i++)
      {
         expectedSize *= matrix.GetDimSize(i);
      }
      ASSERT_EQ(expectedSize, matrix.GetFlatSize());

      // Indexing split tests.
      TestFlatIndices(matrix);
      TestWithAllIndicesHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   static constexpr void TestFlatIndices(mat_t& matrix)
   {
      MatrixUtils::FillSequence(matrix);

      // Test that we actually get a ref.
      matrix.ForEach(
         [&](func_test_t, size_t flatIndex)
         {
            const func_test_t& valueRef = matrix.GetRef(flatIndex);
            ASSERT_EQ(&valueRef, &matrix.GetValues()[flatIndex]);
         });

      // Test that we get the correct values from GetValue
      matrix.ForEach(
         [&matrix](func_test_t value, size_t flatIndex)
         {
            ASSERT_EQ(value, matrix.GetValue(flatIndex));
            ASSERT_EQ(value, matrix[flatIndex]);
         });
   }

   template <size_t... TIdx>
   static constexpr void TestWithAllIndicesHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestWithAllIndices(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   static constexpr void TestWithAllIndices(mat_t& matrix, TIdx...)
   {
      // Test that we actually return refs.
      MatrixUtils::FillSequence(matrix);

      matrix.ForEach(
         [&](func_test_t, TIdx... indices)
         {
            size_t flatIndex = matrix.ToFlatIndex(indices...);
            const func_test_t& valueRef = matrix.GetRef(indices...);
            ASSERT_EQ(&valueRef, &matrix.GetValues()[flatIndex]);
         });

      // Test that we get the correct results from GetValue
      matrix.ForEach(
         [&matrix](func_test_t value, TIdx... indices)
         {
            ASSERT_EQ(value, matrix.GetValue(indices...));
            ASSERT_EQ(value, matrix(indices...));
         });
   }
};

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixMap
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   constexpr static void Test()
   {
      // Create a non-zero default
      mat_t matrix(10);
      TestNoArg(matrix);
      TestOneArg(matrix);
      TestNArgsHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   constexpr static void TestNoArg(mat_t& matrix)
   {
      matrix.Map(
         [](func_test_t& value)
         {
            value = 1;
         });
      ASSERT_TRUE(MatrixUtils::AllEqual(matrix, 1));

      // Test functional map
      MatrixUtils::FillSequence(matrix);
      ASSERT_EQ(MatrixUtils::Sum(matrix), Trianglate(matrix.GetFlatSize() - 1));

      matrix
         .Map(
            [](func_test_t& value)
            {
               value += 1;
            })
         .Map(
            [](func_test_t& value)
            {
               value *= 2;
            });

      // Check that every value was correctly set.
      matrix.ForEach(
         [](func_test_t value, size_t idx)
         {
            ASSERT_EQ(value, (idx + 1) * 2);
         });
   }

   constexpr static void TestOneArg(mat_t& matrix)
   {
      // We really just need to test if it's callable and the indices are correct.
      MatrixUtils::Clear(matrix);
      ASSERT_TRUE(MatrixUtils::AllEqual(matrix, 0));

      matrix.Map(
         [](func_test_t& value, size_t flatIndex)
         {
            value = flatIndex;
         });
      ASSERT_EQ(MatrixUtils::Sum(matrix), Trianglate(matrix.GetFlatSize() - 1));
   }

   template <size_t... TIdx>
   constexpr static void TestNArgsHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestNArgs(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   constexpr static void TestNArgs(mat_t& matrix, TIdx... idx)
   {
      // We really just need to test if it's callable and the indices are correct.
      MatrixUtils::Clear(matrix);
      ASSERT_TRUE(MatrixUtils::AllEqual(matrix, 0));

      // Check that we can iterate with these indices
      matrix.Map(
         [&matrix](func_test_t& value, TIdx... indices)
         {
            value = matrix.ToFlatIndex(indices...);
         });

      ASSERT_EQ(MatrixUtils::Sum(matrix), Trianglate(matrix.GetFlatSize() - 1));

      // Iterate through with flat indices and verify the order is the same, that checks that the
      // indices were correctly iterated and mapped.
      matrix.ForEach(
         [](func_test_t value, size_t idx)
         {
            ASSERT_EQ(value, idx);
         });
   }
};

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixForEach
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   static void Test()
   {
      // Create a non-zero default
      mat_t matrix(10);
      TestNoArg(matrix);
      TestOneArg(matrix);
      TestNArgsHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   static void TestNoArg(mat_t& matrix)
   {
      // Fill with unique values.
      MatrixUtils::FillSequence(matrix);

      // Test that when we get the expected value when iterating and that we iterated the correct
      // number of times. We returned false, but we expect that to be discarded.
      func_test_t lastValue = 0;
      size_t ittCount = 0;

      std::unordered_set<func_test_t> usedValues;

      matrix.ForEach(
         [&](func_test_t value)
         {
            // Make sure we haven't seen the value yet.
            ASSERT_TRUE(usedValues.find(value) == usedValues.end());
            usedValues.insert(value);
            ittCount++;
         });

      ASSERT_EQ(ittCount, matrix.GetFlatSize());
   }

   static void TestOneArg(mat_t& matrix)
   {
      // Fill with unique values.
      MatrixUtils::FillSequence(matrix);

      // Test that when we get the expected value when iterating and that we iterated the correct
      // number of times. We returned false, but we expect that to be discarded.
      func_test_t lastValue = 0;
      size_t ittCount = 0;

      std::unordered_set<func_test_t> usedValues;
      std::unordered_set<size_t> usedIndices;

      matrix.ForEach(
         [&](func_test_t value, size_t flatIndex)
         {
            // Make sure we haven't seen the value yet.
            ASSERT_EQ(value, flatIndex);
            ASSERT_TRUE(usedValues.find(value) == usedValues.end());
            ASSERT_TRUE(usedIndices.find(value) == usedIndices.end());

            usedIndices.insert(flatIndex);
            usedValues.insert(value);
            ittCount++;
         });

      ASSERT_EQ(ittCount, matrix.GetFlatSize());
   }

   template <size_t... TIdx>
   static void TestNArgsHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestNArgs(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   static void TestNArgs(mat_t& matrix, TIdx... idx)
   {
      // Fill with unique values.
      MatrixUtils::FillSequence(matrix);

      // Test that when we get the expected value when iterating and that we iterated the correct
      // number of times. We returned false, but we expect that to be discarded.
      func_test_t lastValue = 0;
      size_t ittCount = 0;

      std::unordered_set<func_test_t> usedValues;
      std::unordered_set<size_t> usedIndices;

      matrix.ForEach(
         [&](func_test_t value, TIdx... indices)
         {
            // Make sure we haven't seen the value yet.
            size_t flatIndex = matrix.ToFlatIndex(indices...);
            ASSERT_EQ(value, flatIndex);
            ASSERT_TRUE(usedValues.find(value) == usedValues.end());
            ASSERT_TRUE(usedIndices.find(value) == usedIndices.end());

            usedIndices.insert(flatIndex);
            usedValues.insert(value);
            ittCount++;
         });

      ASSERT_EQ(ittCount, matrix.GetFlatSize());
   }
};

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixCountIf
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   static void Test()
   {
      // Create a non-zero default
      mat_t matrix(10);
      TestNoArg(matrix);
      TestOneArg(matrix);
      TestNArgsHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   static void TestNoArg(mat_t& matrix)
   {
      matrix.SetValues(10);
      ASSERT_EQ(matrix.CountIf(
                   [](func_test_t value)
                   {
                      return value == 10;
                   }),
         10);

      ASSERT_EQ(matrix.CountIf(
                   [](func_test_t value)
                   {
                      return value == 11;
                   }),
         0);

      matrix.SetValue(11, matrix.GetFlatSize() - 1);
      ASSERT_EQ(matrix.CountIf(
                   [](func_test_t value)
                   {
                      return value == 11;
                   }),
         1);
   }

   static void TestOneArg(mat_t& matrix)
   {
      ASSERT_EQ(matrix.CountIf(
                   [&](func_test_t value, size_t idx)
                   {
                      return idx < matrix.GetFlatSize() - 1;
                   }),
         matrix.GetFlatSize() - 1);
   }

   template <size_t... TIdx>
   static void TestNArgsHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestNArgs(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   static void TestNArgs(mat_t& matrix, TIdx... idx)
   {
      ASSERT_EQ(matrix.CountIf(
                   [&](func_test_t value, TIdx... indices)
                   {
                      size_t flatIndex = matrix.ToFlatIndex(indices...);
                      return flatIndex < matrix.GetFlatSize() - 1;
                   }),
         matrix.GetFlatSize() - 1);
   }
};

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixAllQuery
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   static void Test()
   {
      // Create a non-zero default
      mat_t matrix(10);
      TestNoArg(matrix);
      TestOneArg(matrix);
      TestNArgsHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   static void TestNoArg(mat_t& matrix)
   {
      matrix.SetValues(10);
      size_t numVisited = 0;
      ASSERT_EQ(matrix.All(
                   [&](func_test_t value)
                   {
                      numVisited++;
                      return value == 10;
                   }),
         true);
      ASSERT_EQ(numVisited, matrix.GetFlatSize());

      numVisited = 0;
      ASSERT_EQ(matrix.All(
                   [&](func_test_t value)
                   {
                      numVisited++;
                      return value == 11;
                   }),
         false);
      // Test short circuit.
      ASSERT_EQ(numVisited, 1);

      matrix.SetValue(11, matrix.GetFlatSize() - 1);
      ASSERT_EQ(matrix.All(
                   [&](func_test_t value)
                   {
                      return value == 10;
                   }),
         false);
   }

   static void TestOneArg(mat_t& matrix)
   {
      MatrixUtils::FillSequence(matrix);

      ASSERT_EQ(matrix.All(
                   [&](func_test_t value, size_t idx)
                   {
                      return value == idx;
                   }),
         true);

      ASSERT_EQ(matrix.All(
                   [&](func_test_t value, size_t idx)
                   {
                      return value != idx;
                   }),
         false);
   }

   template <size_t... TIdx>
   static void TestNArgsHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestNArgs(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   static void TestNArgs(mat_t& matrix, TIdx... idx)
   {
      MatrixUtils::FillSequence(matrix);

      ASSERT_EQ(matrix.All(
                   [&](func_test_t value, TIdx... indices)
                   {
                      size_t flatIdx = matrix.ToFlatIndex(indices...);
                      return value == flatIdx;
                   }),
         true);

      ASSERT_EQ(matrix.All(
                   [&](func_test_t value, TIdx... indices)
                   {
                      size_t flatIdx = matrix.ToFlatIndex(indices...);
                      return value != flatIdx;
                   }),
         false);
   }
};

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixAnyQuery
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   static void Test()
   {
      // Create a non-zero default
      mat_t matrix(10);
      TestNoArg(matrix);
      TestOneArg(matrix);
      TestNArgsHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   static void TestNoArg(mat_t& matrix)
   {
      matrix.SetValues(10);
      size_t numVisited = 0;
      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value)
                   {
                      numVisited++;
                      return value == 10;
                   }),
         true);
      // Test short circuit.
      ASSERT_EQ(numVisited, 1);

      numVisited = 0;
      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value)
                   {
                      numVisited++;
                      return value == 11;
                   }),
         false);
      // Test short circuit.
      ASSERT_EQ(numVisited, matrix.GetFlatSize());

      matrix.SetValue(11, matrix.GetFlatSize() - 1);
      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value)
                   {
                      return value == 11;
                   }),
         true);
   }

   static void TestOneArg(mat_t& matrix)
   {
      MatrixUtils::FillSequence(matrix);

      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value, size_t idx)
                   {
                      return value == idx;
                   }),
         true);

      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value, size_t idx)
                   {
                      return value != idx;
                   }),
         false);
   }

   template <size_t... TIdx>
   static void TestNArgsHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestNArgs(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   static void TestNArgs(mat_t& matrix, TIdx... idx)
   {
      MatrixUtils::FillSequence(matrix);

      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value, TIdx... indices)
                   {
                      size_t flatIdx = matrix.ToFlatIndex(indices...);
                      return value == flatIdx;
                   }),
         true);

      ASSERT_EQ(matrix.Any(
                   [&](func_test_t value, TIdx... indices)
                   {
                      size_t flatIdx = matrix.ToFlatIndex(indices...);
                      return value != flatIdx;
                   }),
         false);
   }
};

template <template <typename, size_t...> typename TMatrix, size_t... TDims> class TestMatrixFold
{
public:
   using mat_t = TMatrix<func_test_t, TDims...>;

   static void Test()
   {
      // Create a non-zero default
      mat_t matrix(10);

      TestNoArg(matrix);
      TestOneArg(matrix);
      TestNArgsHelper(matrix, std::make_index_sequence<sizeof...(TDims)>{});
   }

private:
   static void TestFoldResult(const std::vector<func_test_t>& result)
   {
      for (size_t i = 0; i < result.size(); i++)
      {
         ASSERT_EQ(i, result[i]);
      }
   }

   static void TestNoArg(mat_t& matrix)
   {
      MatrixUtils::FillSequence(matrix);
      size_t sum = 0;
      matrix.Fold(sum,
         [](auto& acc, func_test_t value)
         {
            acc += value;
         });
      ASSERT_EQ(sum, Trianglate(matrix.GetFlatSize() - 1));

      // Test on generic result
      std::vector<func_test_t> foldResult;
      matrix.Fold(foldResult,
         [](auto& acc, func_test_t value)
         {
            acc.push_back(value);
         });
      TestFoldResult(foldResult);
   }

   static void TestOneArg(mat_t& matrix)
   {
      // Test on generic result
      std::vector<func_test_t> foldResult;
      matrix.Fold(foldResult,
         [](auto& acc, func_test_t value, size_t flatIdx)
         {
            acc.push_back(flatIdx);
         });
      TestFoldResult(foldResult);
   }

   template <size_t... TIdx>
   static void TestNArgsHelper(mat_t& matrix, std::index_sequence<TIdx...>)
   {
      TestNArgs(matrix, TIdx...);
   }

   template <typename... TIdx>
      requires(sizeof...(TDims) == sizeof...(TIdx))
   static void TestNArgs(mat_t& matrix, TIdx... idx)
   {
      // Test on generic result
      std::vector<func_test_t> foldResult;
      matrix.Fold(foldResult,
         [&](auto& acc, func_test_t value, TIdx... indices)
         {
            size_t flatIndex = matrix.ToFlatIndex(indices...);
            acc.push_back(flatIndex);
         });
      TestFoldResult(foldResult);
   }
};

class TestMatrixFindIf
{
public:
   static void Test()
   {
      size_t idx;
      size_t x, y, z;
      bool found = false;

      StaticVectorState<func_test_t, 10> vec(0);
      MatrixUtils::FillSequence(vec);

      // Find the first and last elements.
      found = vec.FindIf(
         [](func_test_t value)
         {
            return value == 0;
         },
         idx);
      ASSERT_EQ(idx, 0);
      ASSERT_TRUE(found);

      found = vec.FindIf(
         [&vec](func_test_t value, size_t flatIdx)
         {
            return value == vec.GetFlatSize() - 1;
         },
         idx);
      ASSERT_EQ(idx, vec.GetFlatSize() - 1);
      ASSERT_TRUE(found);

      // Test on a matrix and test each combination of parameters.
      HeapMatrixState<func_test_t, 10, 10, 10> matrix(0);
      MatrixUtils::FillSequence(matrix);

      // Find without taking in any indices.
      found = matrix.FindIf(
         [](func_test_t value)
         {
            return value == 0;
         },
         idx);
      ASSERT_EQ(idx, 0);
      ASSERT_TRUE(found);

      found = matrix.FindIf(
         [&matrix](func_test_t value)
         {
            return value == matrix.GetFlatSize() - 1;
         },
         x, y, z);
      ASSERT_EQ(matrix.ToFlatIndex(x, y, z), matrix.GetFlatSize() - 1);
      ASSERT_TRUE(found);

      // Test taking in flat index
      found = matrix.FindIf(
         [&matrix](func_test_t value, size_t flatIndex)
         {
            return value == matrix.GetFlatSize() - 1;
         },
         idx);
      ASSERT_EQ(idx, matrix.GetFlatSize() - 1);
      ASSERT_TRUE(found);

      found = matrix.FindIf(
         [](func_test_t value, size_t flatIndex)
         {
            return value == 0;
         },
         x, y, z);
      ASSERT_EQ(matrix.ToFlatIndex(x, y, z), 0);
      ASSERT_TRUE(found);

      // Test taking in all indices
      found = matrix.FindIf(
         [](func_test_t value, size_t i, size_t j, size_t k)
         {
            return value == 0;
         },
         x, y, z);
      ASSERT_EQ(matrix.ToFlatIndex(x, y, z), 0);
      ASSERT_TRUE(found);

      found = matrix.FindIf(
         [&matrix](func_test_t value, size_t i, size_t j, size_t k)
         {
            return value == matrix.GetFlatSize() - 1;
         },
         idx);
      ASSERT_EQ(idx, matrix.GetFlatSize() - 1);
      ASSERT_TRUE(found);

      // Test not found.
      found = matrix.FindIf(
         [&matrix](func_test_t value, size_t i, size_t j, size_t k)
         {
            return false;
         },
         idx);
      ASSERT_EQ(idx, matrix.GetFlatSize() - 1);
      ASSERT_FALSE(found);
   }
};

TEST(StateTests, TestMatrixQueries)
{
   TestMatrixQueries<HeapVectorState, 10>::Test();
   TestMatrixQueries<HeapMatrixState, 1, 2>::Test();
   TestMatrixQueries<StaticVectorState, 10>::Test();
   TestMatrixQueries<StaticMatrixState, 1, 2>::Test();
}

TEST(StateTests, TestMatrixForEach)
{
   TestMatrixForEach<HeapVectorState, 10>::Test();
   TestMatrixForEach<HeapMatrixState, 5, 1, 2>::Test();
   TestMatrixForEach<StaticVectorState, 10>::Test();
   TestMatrixForEach<StaticMatrixState, 5, 1, 2>::Test();
}

TEST(StateTests, TestMatrixCountIf)
{
   TestMatrixCountIf<HeapVectorState, 10>::Test();
   TestMatrixCountIf<HeapMatrixState, 5, 1, 2>::Test();
   TestMatrixCountIf<StaticVectorState, 10>::Test();
   TestMatrixCountIf<StaticMatrixState, 5, 1, 2>::Test();
}

TEST(StateTests, TestMatrixAllQuery)
{
   TestMatrixAllQuery<HeapVectorState, 10>::Test();
   TestMatrixAllQuery<HeapMatrixState, 5, 1, 2>::Test();
   TestMatrixAllQuery<StaticVectorState, 10>::Test();
   TestMatrixAllQuery<StaticMatrixState, 5, 1, 2>::Test();
}

TEST(StateTests, TestMatrixAnyQuery)
{
   TestMatrixAnyQuery<HeapVectorState, 10>::Test();
   TestMatrixAnyQuery<HeapMatrixState, 5, 1, 2>::Test();
   TestMatrixAnyQuery<StaticVectorState, 10>::Test();
   TestMatrixAnyQuery<StaticMatrixState, 5, 1, 2>::Test();
}

TEST(StateTests, TestMatrixMap)
{
   TestMatrixMap<HeapVectorState, 10>::Test();
   TestMatrixMap<HeapMatrixState, 5, 1, 2>::Test();
   TestMatrixMap<StaticVectorState, 10>::Test();
   TestMatrixMap<StaticMatrixState, 5, 1, 2>::Test();
}

TEST(StateTests, TestMatrixFold)
{

   TestMatrixFold<HeapVectorState, 10>::Test();
   TestMatrixFold<HeapMatrixState, 5, 1, 2>::Test();
   TestMatrixFold<StaticVectorState, 10>::Test();
   TestMatrixFold<StaticMatrixState, 5, 1, 2>::Test();

}

TEST(StateTests, TestMatrixFindIf)
{
   TestMatrixFindIf::Test();
}

TEST(StateTests, TestIndexConversions)
{
   // Edge cases
   ASSERT_EQ(ComputeFlatIndex<10>(0), 0);
   ASSERT_EQ((ComputeFlatIndex<10, 10>(0, 0)), 0);
   ASSERT_EQ((ComputeFlatIndex<10, 10, 10>(0, 0, 0)), 0);
   ASSERT_EQ((ComputeFlatIndex<10, 10, 10, 10>(0, 0, 0, 0)), 0);

   // Last index
   ASSERT_EQ(ComputeFlatIndex<10>(9), 9);
   ASSERT_EQ((ComputeFlatIndex<10, 10>(9, 9)), 99);
   ASSERT_EQ((ComputeFlatIndex<5, 2, 2>(4, 1, 1)), 19);

   // Just test a few random conversions and verify that the result is correct
   ASSERT_EQ(ComputeFlatIndex<10>(5), 5);
   ASSERT_EQ(ComputeFlatIndex<10>(11), 11);

   ASSERT_EQ((ComputeFlatIndex<10, 10>(5, 0)), 50);
   ASSERT_EQ((ComputeFlatIndex<10, 10>(2, 2)), 22);

   ASSERT_EQ((ComputeFlatIndex<5, 2, 2>(3, 0, 2)), 14);
   ASSERT_EQ((ComputeFlatIndex<2, 5, 9>(2, 2, 2)), 110);
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
