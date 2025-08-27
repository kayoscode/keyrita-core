//
// 2017-2024
//

#pragma once

#include "cpp_events/Event.h"
#include <chrono>

namespace kc
{
/**
 * Class to handle time differentials
 * Construct the timer object with no parameters which will automatically reset the data
 * calling any of the functions nanoseconds(), microseconds(), milliseconds(), seconds() will return
 * the time passed since the last call of reset calling reset clears the data in the timer
 * 2017-2024
 * */
class ThreadSafeTimer
{
public:
   ThreadSafeTimer()
   {
      Reset();
   }

   ThreadSafeTimer(const ThreadSafeTimer& other) : mPrevTimePoint(other.mPrevTimePoint)
   {
   }

   ThreadSafeTimer& operator=(const ThreadSafeTimer& other)
   {
      auto lock(mMutexHelper.GetWriteLock());
      mPrevTimePoint = other.mPrevTimePoint;
      return *this;
   }

   ~ThreadSafeTimer() = default;

   void Reset()
   {
      auto lock(mMutexHelper.GetWriteLock());
      mPrevTimePoint = std::chrono::high_resolution_clock::now();
   }

   [[nodiscard]] uint64_t Nanoseconds() const
   {
      auto lock(mMutexHelper.GetReadLock());
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::nanoseconds>(now - mPrevTimePoint).count());
   }

   [[nodiscard]] uint64_t Microseconds() const
   {
      auto lock(mMutexHelper.GetReadLock());
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::microseconds>(now - mPrevTimePoint).count());
   }

   [[nodiscard]] uint64_t Milliseconds() const
   {
      auto lock(mMutexHelper.GetReadLock());
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::milliseconds>(now - mPrevTimePoint).count());
   }

   [[nodiscard]] uint64_t Seconds() const
   {
      auto lock(mMutexHelper.GetReadLock());
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::seconds>(now - mPrevTimePoint).count());
   }

private:
   std::chrono::high_resolution_clock::time_point mPrevTimePoint;
   MutexHelper mMutexHelper;
};

/**
 * The lower overhead, non-thread safe implementation of the timer.
 * 2017-2024
 * */
class Timer
{
public:
   Timer()
   {
      Reset();
   }

   ~Timer() = default;

   void Reset()
   {
      mPrevTimePoint = std::chrono::high_resolution_clock::now();
   }

   [[nodiscard]] uint64_t Nanoseconds() const
   {
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::nanoseconds>(now - mPrevTimePoint).count());
   }

   [[nodiscard]] uint64_t Microseconds() const
   {
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::microseconds>(now - mPrevTimePoint).count());
   }

   [[nodiscard]] uint64_t Milliseconds() const
   {
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::milliseconds>(now - mPrevTimePoint).count());
   }

   [[nodiscard]] uint64_t Seconds() const
   {
      const auto now = std::chrono::high_resolution_clock::now();
      return static_cast<uint64_t>(
         std::chrono::duration_cast<std::chrono::seconds>(now - mPrevTimePoint).count());
   }

private:
   std::chrono::high_resolution_clock::time_point mPrevTimePoint;
};
}