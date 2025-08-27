#pragma once

#include "keyrita_core/MatrixQuery.h"

#include <array>
#include <concepts>
#include <span>
#include <type_traits>

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
template <template <typename T, size_t... TDims> class TAlloc, typename T, size_t... TDims>
concept MatrixAlloc = ScalarStateValue<T> && std::is_default_constructible_v<T> &&
                      requires(TAlloc<T, TDims...>& alloc) {
                         { alloc.GetVec() } -> std::same_as<std::span<T, TotalVecSize<TDims...>()>>;
                         { alloc.GetVec() } -> std::convertible_to<std::span<const T, TotalVecSize<TDims...>()>>;
                      };

/**
 * @brief      Allocates a buffer of static memory.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixStaticAlloc
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
   std::span<T, TotalVecSize<TDims...>()> GetVec()
   {
      return mData;
   }

private:
   std::array<T, TotalVecSize<TDims...>()> mData;
};

/**
 * @brief      Allocates a buffer of heap memory.
 */
template <ScalarStateValue T, size_t... TDims> class MatrixHeapAlloc
{
public:
   MatrixHeapAlloc()
   {
      mData = new std::array<T, TotalVecSize<TDims...>()>();
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
   std::span<T, TotalVecSize<TDims...>()> GetVec()
   {
      return *mData;
   }

private:
   // Raw pointer since the logic in this class is self contained and we want to stay efficient,
   // even in debug builds
   std::array<T, TotalVecSize<TDims...>()>* mData;
};
}   // namespace kc