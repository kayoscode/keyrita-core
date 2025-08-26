#pragma once

#include "keyrita_core/State.h"

#include <array>
#include <concepts>
#include <span>

namespace kc
{
/**
 * @brief      Concept validating a type fulfills the contract of a matrix allocator
 * An allocator is defined by a class that can return a vector of the required size by calling
 * GetVec()
 *
 * @tparam     T          The type returned by the allocator.
 * @tparam     TFlatSize  The flat size of the vector returned by the allocator class.
 */
template <template <typename T, size_t TFlatSize> class TAlloc, typename T, size_t TFlatSize>
concept MatrixAlloc = ScalarStateValue<T> && requires(TAlloc<T, TFlatSize>& alloc) {
   { alloc.GetVec() } -> std::same_as<std::span<T, TFlatSize>>;
   { alloc.GetVec() } -> std::convertible_to<std::span<const T, TFlatSize>>;
};

/**
 * @brief      Allocates a buffer of static memory.
 */
template <ScalarStateValue T, size_t TFlatSize> class MatrixStaticAlloc
{
public:
   MatrixStaticAlloc()
   {
   }

   // No destructor needed for static data.

   /**
    * @brief      Api call needed to fulfill the contract of the allocator.
    * @return     The raw data.
    */
   std::span<T, TFlatSize> GetVec()
   {
      return mData;
   }

private:
   std::array<T, TFlatSize> mData;
};

/**
 * @brief      Allocates a buffer of heap memory.
 */
template <ScalarStateValue T, size_t TFlatSize> class MatrixHeapAlloc
{
public:
   MatrixHeapAlloc()
   {
      mData = new std::array<T, TFlatSize>();
   }

   virtual ~MatrixHeapAlloc()
   {
      delete mData;
   }

   MatrixHeapAlloc(const MatrixHeapAlloc& other) = delete;
   MatrixHeapAlloc(MatrixHeapAlloc&& other) = delete;
   MatrixHeapAlloc& operator=(const MatrixHeapAlloc& other) = delete;

   // No destructor needed for static data.

   /**
    * @brief      Api call needed to fulfill the contract of the allocator.
    * @return     The raw data.
    */
   std::span<T, TFlatSize> GetVec()
   {
      return *mData;
   }

private:
   // Raw pointer since the logic in this class is self contained and we want to stay efficient,
   // even in debug builds
   std::array<T, TFlatSize>* mData;
};
}   // namespace kc