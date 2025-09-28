#pragma once

#include "keyrita_core/State/MatrixState.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"
#include <cstdint>
#include <cstdlib>
#include <malloc/_malloc_type.h>

/**
 * @brief      Base class for a dataset containing corpus information for a specific alphabet.
 *             Stores N-grams to depth 3
 * @TParam TAlphabetSize   The max number of characters available in the alphabet.
 */
namespace kc
{
/**
 * @brief      Provides the readonly interface to unigrams. Stores a matrix of N character
 * frequencies.
 * @tparam     TAlphabetSize  The number of characters in the alphabet being used.
 */
template <int TAlphabetSize> class IUnigrams
{
public:
   IUnigrams() = default;
   virtual ~IUnigrams() = default;

   int GetAlphabetSize() const
   {
      return TAlphabetSize;
   }

   const IMatrixState<uint64_t, TAlphabetSize>& GetUnigramsMatrix() const
   {
      return mUnigrams;
   }

   uint64_t GetUnigramFreq(int alphabetIndex) const
   {
      return mUnigrams(alphabetIndex);
   }

protected:
   HeapMatrixState<uint64_t, TAlphabetSize> mUnigrams;
};

template <int TAlphabetSize> class Unigrams : public IUnigrams<TAlphabetSize>
{
public:
   HeapMatrixState<uint64_t, TAlphabetSize>& GetUnigramsMatrix()
   {
      return this->mUnigrams;
   }

   void SetUnigramFreq(int alphabetIndex, size_t frequency)
   {
      this->mUnigrams.GetRef(alphabetIndex) = frequency;
   }

private:
};

/**
 * @brief      Stores a matrix representing the bigram frequencies loaded into a dataset.
 * @tparam     TAlphabetSize  The number of characters in the alphabet.
 */
template <int TAlphabetSize> class IBigrams
{
public:
   IBigrams() = default;
   virtual ~IBigrams() = default;

   int GetAlphabetSize() const
   {
      return TAlphabetSize;
   }

   uint64_t GetBigramFreq(int alphabetIdx1, int alphabetIdx2) const
   {
      return mBigrams(alphabetIdx1, alphabetIdx2);
   }

   const IMatrixState<uint64_t, TAlphabetSize, TAlphabetSize>& GetBigramsMatrix() const
   {
      return mBigrams;
   }

   /**
    * @brief      Computes the frequencies of the unigrams starting with the given character
    * and returns it wrapped in a view. Note the view is a copy. The existing data in the matrix
    * will not be preserved.
    *
    * @param[in]  alphabetIdx   The index in the alphabet to use as the starting character.
    * @param      result        The unigram frequencies of the bigrams starting with the first
    * character.
    */
   void GetUnigramsStartingWith(int alphabetIdx1, Unigrams<TAlphabetSize>& result)
   {
      auto& resultMatrix = result.GetUnigramsMatrix();
      const auto& bigramsMatrix = GetBigramsMatrix();

      resultMatrix.Map(resultMatrix,
               [&bigramsMatrix, alphabetIdx1](uint64_t& result, uint64_t value, size_t alphabetIdx2)
               {
                  result = bigramsMatrix.GetValue(alphabetIdx1, alphabetIdx2);
               });
   }

   /**
    * @brief      Computes the frequencies of the unigrams ending with the given character
    * and returns it wrapped in a view. Note the view is a copy. The existing data in the matrix
    * will not be preserved.
    *
    * @param[in]  alphabetIdx   The index in the alphabet to use as the starting character.
    * @param      result        The unigram frequencies of the bigrams starting with the first
    * character.
    */
   void GetUnigramsEndingWith(int alphabetIdx1, Unigrams<TAlphabetSize>& result)
   {
      auto& resultMatrix = result.GetUnigramsMatrix();
      const auto& bigramsMatrix = GetBigramsMatrix();

      resultMatrix.Map(resultMatrix,
               [&bigramsMatrix, alphabetIdx1](uint64_t& result, uint64_t value, size_t alphabetIdx2)
               {
                  result = bigramsMatrix.GetValue(alphabetIdx2, alphabetIdx1);
               });
   }

public:
   HeapMatrixState<uint64_t, TAlphabetSize, TAlphabetSize> mBigrams;
};

/**
 * @brief      Stores a matrix representing the bigram frequencies loaded into a dataset.
 * @tparam     TAlphabetSize  The number of characters in the alphabet.
 */
template <int TAlphabetSize> class Bigrams : public IBigrams<TAlphabetSize>
{
public:
   HeapMatrixState<uint64_t, TAlphabetSize, TAlphabetSize>& GetBigramsMatrix()
   {
      return this->mBigrams;
   }

   void SetBigramFrequency(int alphabetIdx1, int alphabetIdx2, size_t frequency)
   {
      this->mBigrams.GetRef(alphabetIdx1, alphabetIdx2) = frequency;
   }

private:
};

/**
 * @brief      Stores a matrix representing the trigram frequencies loaded into a dataset.
 * @tparam     TAlphabetSize  The number of characters in the alphabet.
 */
template <int TAlphabetSize> class ITrigrams
{
public:
   ITrigrams() = default;
   virtual ~ITrigrams() = default;

   int GetAlphabetSize() const
   {
      return TAlphabetSize;
   }

   uint64_t GetTrigramFrequency(int alphabetIdx1, int alphabetIdx2, int alphabetIdx3) const
   {
      return mTrigrams(alphabetIdx1, alphabetIdx2, alphabetIdx3);
   }

   const IMatrixState<uint64_t, TAlphabetSize, TAlphabetSize, TAlphabetSize>& GetTrigramsMatrix() const
   {
      return mTrigrams;
   }

protected:

   HeapMatrixState<uint64_t, TAlphabetSize, TAlphabetSize, TAlphabetSize> mTrigrams;
};

/**
 * @brief      Stores a matrix representing the trigram frequencies loaded into a dataset.
 * @tparam     TAlphabetSize  The number of characters in the alphabet.
 */
template <int TAlphabetSize> class Trigrams : public ITrigrams<TAlphabetSize>
{
public:
   HeapMatrixState<uint64_t, TAlphabetSize, TAlphabetSize, TAlphabetSize>& GetTrigramsMatrix()
   {
      return this->mTrigrams;
   }

   void SetTrigramFrequency(int alphabetIdx1, int alphabetIdx2, int alphabetIdx3, uint64_t frequency)
   {
      this->mTrigrams(alphabetIdx1, alphabetIdx2, alphabetIdx3) = frequency;
   }

private:
};

/**
 * Stores the skipgram frequencies
 * The skipgrams are as a vector of bigrams where each the first index represents the
 * skipgram order, and the bigram frequency following represents the frequnecy of that bigram
 * such that skipgrams[2] -> ds[0] xx ds[3] where the number of characters between is equal to i
 * where we are using skipgrams[i]
 *
 * Example: skipgrams[0] = 0 characters between
 *          skipgrams[3] = 3 characters between
 *          etc
 */
template <int TAlphabetSize, int TNumSkipgrams> class ISkipgrams
{
public:
   ISkipgrams() = default;
   virtual ~ISkipgrams() = default;

   int GetAlphabetSize() const
   {
      return TAlphabetSize;
   }

   int GetNumSkipgrams() const
   {
      return TNumSkipgrams;
   }

   const IMatrixState<uint64_t, TNumSkipgrams, TAlphabetSize, TAlphabetSize>& GetSkipgramsMatrix() const
   {
      return mSkipgrams;
   }

   uint64_t GetSkipgramFrequency(int skipgramLength, int alphabetIdx1, int alphabetIdx2) const
   {
      return mSkipgrams(skipgramLength, alphabetIdx1, alphabetIdx2);
   }

public:
   HeapMatrixState<uint64_t, TNumSkipgrams, TAlphabetSize, TAlphabetSize> mSkipgrams;
};

template <int TAlphabetSize, int TNumSkipgrams>
class Skipgrams : public ISkipgrams<TAlphabetSize, TNumSkipgrams>
{
public:
   HeapMatrixState<uint64_t, TNumSkipgrams, TAlphabetSize, TAlphabetSize>& GetSkipgramsMatrix()
   {
      return this->mSkipgrams;
   }

   void SetSkipgramFrequency(int skipgramLength, int alphabetIdx1, int alphabetIdx2, uint64_t frequency)
   {
      this->mSkipgrams.GetRef(skipgramLength, alphabetIdx1, alphabetIdx2) = frequency;
   }

private:
};

template <int TAlphabetSize, int TNumSkipgrams> class Dataset
{
public:
   Dataset() = default;
   virtual ~Dataset() = default;

   /**
    * @return     The matrix of unigrams.
    */
   const IUnigrams<TAlphabetSize>& GetUnigrams() const
   {
      return mUnigrams;
   }

   /**
    * @return     The matrix of bigrams.
    */
   const IBigrams<TAlphabetSize>& GetBigrams() const
   {
      return mBigrams;
   }

   /**
    * @return     The matrix of trigrams.
    */
   const ITrigrams<TAlphabetSize>& GetTrigrams() const
   {
      return mTrigrams;
   }

   /**
    * @return     The matrix of trigrams.
    */
   const ISkipgrams<TAlphabetSize, TNumSkipgrams>& GetSkipgrams() const
   {
      return mSkipgrams;
   }

   /**
    * @return     The number of characters in the alphabet.
    */
   int GetAlphabetSize() const
   {
      return TAlphabetSize;
   }

private:
   /**
    * Stores the character frequencies for each letter of the alphabet.
    */
   Unigrams<TAlphabetSize> mUnigrams;
   Bigrams<TAlphabetSize> mBigrams;
   Trigrams<TAlphabetSize> mTrigrams;
   Skipgrams<TAlphabetSize, TNumSkipgrams> mSkipgrams;
};
}   // namespace kc