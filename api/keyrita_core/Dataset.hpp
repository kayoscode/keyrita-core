#pragma once

#include "keyrita_core/State/MatrixState.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"
#include <cstdint>
#include <cstdlib>
#include <malloc/_malloc_type.h>
#include <memory>

/**
 * @brief      Base class for a dataset containing corpus information for a specific char.
 *             Stores N-grams to depth 3
 * @TParam TNumCharacters   The max number of characters available in the char.
 */
namespace kc
{
/**
 * @brief      Provides the readonly interface to unigrams. Stores a matrix of N character
 * frequencies.
 * @tparam     TNumCharacters  The number of characters in the char being used.
 */
template <int TNumCharacters> class IUnigrams
{
public:
   IUnigrams() = default;
   virtual ~IUnigrams() = default;

   int GetNumCharacters() const
   {
      return TNumCharacters;
   }

   const IMatrixState<uint64_t, TNumCharacters>& GetUnigramsMatrix() const
   {
      return mUnigrams;
   }

   uint64_t GetUnigramFreq(int charIndex) const
   {
      return mUnigrams(charIndex);
   }

protected:
   HeapMatrixState<uint64_t, TNumCharacters> mUnigrams;
};

template <int TNumCharacters> class Unigrams : public IUnigrams<TNumCharacters>
{
public:
   HeapMatrixState<uint64_t, TNumCharacters>& GetUnigramsMatrix()
   {
      return this->mUnigrams;
   }

   void Clear()
   {
      MatrixUtils::Clear(this->mUnigrams);
   }

   void SetUnigramFreq(int charIndex, size_t frequency)
   {
      this->mUnigrams.GetRef(charIndex) = frequency;
   }

private:
};

/**
 * @brief      Stores a matrix representing the bigram frequencies loaded into a dataset.
 * @tparam     TNumCharacters  The number of characters in the char.
 */
template <int TNumCharacters> class IBigrams
{
public:
   IBigrams() = default;
   virtual ~IBigrams() = default;

   int GetNumCharacters() const
   {
      return TNumCharacters;
   }

   uint64_t GetBigramFreq(int charIdx1, int charIdx2) const
   {
      return mBigrams(charIdx1, charIdx2);
   }

   const IMatrixState<uint64_t, TNumCharacters, TNumCharacters>& GetBigramsMatrix() const
   {
      return mBigrams;
   }

   /**
    * @brief      Computes the frequencies of the unigrams starting with the given character
    * and returns it wrapped in a view. Note the view is a copy. The existing data in the matrix
    * will not be preserved.
    *
    * @param[in]  charIdx   The index in the char to use as the starting character.
    * @param      result        The unigram frequencies of the bigrams starting with the first
    * character.
    */
   void GetUnigramsStartingWith(int charIdx1, Unigrams<TNumCharacters>& result)
   {
      auto& resultMatrix = result.GetUnigramsMatrix();
      const auto& bigramsMatrix = GetBigramsMatrix();

      resultMatrix.Map(resultMatrix,
         [&bigramsMatrix, charIdx1](uint64_t& result, uint64_t value, size_t charIdx2)
         {
            result = bigramsMatrix.GetValue(charIdx1, charIdx2);
         });
   }

   /**
    * @brief      Computes the frequencies of the unigrams ending with the given character
    * and returns it wrapped in a view. Note the view is a copy. The existing data in the matrix
    * will not be preserved.
    *
    * @param[in]  charIdx   The index in the char to use as the starting character.
    * @param      result        The unigram frequencies of the bigrams starting with the first
    * character.
    */
   void GetUnigramsEndingWith(int charIdx1, Unigrams<TNumCharacters>& result)
   {
      auto& resultMatrix = result.GetUnigramsMatrix();
      const auto& bigramsMatrix = GetBigramsMatrix();

      resultMatrix.Map(resultMatrix,
         [&bigramsMatrix, charIdx1](uint64_t& result, uint64_t value, size_t charIdx2)
         {
            result = bigramsMatrix.GetValue(charIdx2, charIdx1);
         });
   }

public:
   HeapMatrixState<uint64_t, TNumCharacters, TNumCharacters> mBigrams;
};

/**
 * @brief      Stores a matrix representing the bigram frequencies loaded into a dataset.
 * @tparam     TNumCharacters  The number of characters in the char.
 */
template <int TNumCharacters> class Bigrams : public IBigrams<TNumCharacters>
{
public:
   void Clear()
   {
      MatrixUtils::Clear(this->mBigrams);
   }

   HeapMatrixState<uint64_t, TNumCharacters, TNumCharacters>& GetBigramsMatrix()
   {
      return this->mBigrams;
   }

   void SetBigramFrequency(int charIdx1, int charIdx2, size_t frequency)
   {
      this->mBigrams.GetRef(charIdx1, charIdx2) = frequency;
   }

private:
};

/**
 * @brief      Stores a matrix representing the trigram frequencies loaded into a dataset.
 * @tparam     TNumCharacters  The number of characters in the char.
 */
template <int TNumCharacters> class ITrigrams
{
public:
   ITrigrams() = default;
   virtual ~ITrigrams() = default;

   int GetNumCharacters() const
   {
      return TNumCharacters;
   }

   uint64_t GetTrigramFrequency(int charIdx1, int charIdx2, int charIdx3) const
   {
      return mTrigrams(charIdx1, charIdx2, charIdx3);
   }

   const IMatrixState<uint64_t, TNumCharacters, TNumCharacters, TNumCharacters>&
   GetTrigramsMatrix() const
   {
      return mTrigrams;
   }

protected:
   HeapMatrixState<uint64_t, TNumCharacters, TNumCharacters, TNumCharacters> mTrigrams;
};

/**
 * @brief      Stores a matrix representing the trigram frequencies loaded into a dataset.
 * @tparam     TNumCharacters  The number of characters in the char.
 */
template <int TNumCharacters> class Trigrams : public ITrigrams<TNumCharacters>
{
public:
   void Clear()
   {
      MatrixUtils::Clear(this->mTrigrams);
   }

   HeapMatrixState<uint64_t, TNumCharacters, TNumCharacters, TNumCharacters>& GetTrigramsMatrix()
   {
      return this->mTrigrams;
   }

   void SetTrigramFrequency(
      int charIdx1, int charIdx2, int charIdx3, uint64_t frequency)
   {
      this->mTrigrams(charIdx1, charIdx2, charIdx3) = frequency;
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
template <int TNumCharacters, int TNumSkipgrams> class ISkipgrams
{
public:
   ISkipgrams() = default;
   virtual ~ISkipgrams() = default;

   int GetNumCharacters() const
   {
      return TNumCharacters;
   }

   int GetNumSkipgrams() const
   {
      return TNumSkipgrams;
   }

   const IMatrixState<uint64_t, TNumSkipgrams, TNumCharacters, TNumCharacters>&
   GetSkipgramsMatrix() const
   {
      return mSkipgrams;
   }

   uint64_t GetSkipgramFrequency(int skipgramLength, int charIdx1, int charIdx2) const
   {
      return mSkipgrams(skipgramLength, charIdx1, charIdx2);
   }

public:
   HeapMatrixState<uint64_t, TNumSkipgrams, TNumCharacters, TNumCharacters> mSkipgrams;
};

template <int TNumCharacters, int TNumSkipgrams>
class Skipgrams : public ISkipgrams<TNumCharacters, TNumSkipgrams>
{
public:
   void Clear()
   {
      MatrixUtils::Clear(this->mSkipgrams);
   }

   HeapMatrixState<uint64_t, TNumSkipgrams, TNumCharacters, TNumCharacters>& GetSkipgramsMatrix()
   {
      return this->mSkipgrams;
   }

   void SetSkipgramFrequency(
      int skipgramLength, int charIdx1, int charIdx2, uint64_t frequency)
   {
      this->mSkipgrams.GetRef(skipgramLength, charIdx1, charIdx2) = frequency;
   }

private:
};

/**
 * @brief      Defines a readonly view of a dataset. Provides unigrams, bigrams, trigrams and
 * skipgrams
 *
 * @tparam     TNumCharacters  The number of characters allowed by the dataset.
 * @tparam     TNumSkipgrams   The number of skipgrams the dataset cares about.
 */
template <int TNumCharacters, int TNumSkipgrams> class IDataset
{
public:
   IDataset() = default;
   virtual ~IDataset() = default;

   /**
    * @return     The matrix of unigrams.
    */
   const IUnigrams<TNumCharacters>& GetUnigrams() const
   {
      return mUnigrams;
   }

   /**
    * @return     The matrix of bigrams.
    */
   const IBigrams<TNumCharacters>& GetBigrams() const
   {
      return mBigrams;
   }

   /**
    * @return     The matrix of trigrams.
    */
   const ITrigrams<TNumCharacters>& GetTrigrams() const
   {
      return mTrigrams;
   }

   /**
    * @return     The matrix of trigrams.
    */
   const ISkipgrams<TNumCharacters, TNumSkipgrams>& GetSkipgrams() const
   {
      return mSkipgrams;
   }

   /**
    * @return     The number of characters in the char.
    */
   constexpr int GetNumCharacters() const
   {
      return TNumCharacters;
   }

protected:
   /**
    * Stores the character frequencies for each letter of the char.
    */
   Unigrams<TNumCharacters> mUnigrams;
   Bigrams<TNumCharacters> mBigrams;
   Trigrams<TNumCharacters> mTrigrams;
   Skipgrams<TNumCharacters, TNumSkipgrams> mSkipgrams;
};

/**
 * @brief      Defines a writable queriable dataset for all loaded data about the language.
 */
template <int TNumCharacters, int TNumSkipgrams>
class Dataset : public IDataset<TNumCharacters, TNumSkipgrams>
{
public:
   Dataset() = default;
   virtual ~Dataset() = default;

   /**
    * @return     The matrix of unigrams.
    */
   Unigrams<TNumCharacters>& GetUnigrams()
   {
      return this->mUnigrams;
   }

   /**
    * @return     The matrix of bigrams.
    */
   Bigrams<TNumCharacters>& GetBigrams()
   {
      return this->mBigrams;
   }

   /**
    * @return     The matrix of trigrams.
    */
   Trigrams<TNumCharacters>& GetTrigrams()
   {
      return this->mTrigrams;
   }

   /**
    * @return     The matrix of trigrams.
    */
   Skipgrams<TNumCharacters, TNumSkipgrams>& GetSkipgrams()
   {
      return this->mSkipgrams;
   }

   void Clear()
   {
      this->mUnigrams.Clear();
      this->mBigrams.Clear();
      this->mTrigrams.Clear();
      this->mSkipgrams.Clear();
   }
};
}   // namespace kc