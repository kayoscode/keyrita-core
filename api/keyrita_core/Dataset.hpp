#pragma once

#include "keyrita_core/State/MatrixState.hpp"
#include "keyrita_core/State/MatrixUtils.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

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

   const IMatrixState<uint64_t, TNumCharacters>& GetMatrix() const
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
   void Clear()
   {
      MatrixUtils::Clear(this->mUnigrams);
   }

   void CopyFrom(const IMatrixState<uint64_t, TNumCharacters>& unigramData)
   {
      this->mUnigrams.Zip(this->mUnigrams, unigramData,
         [](uint64_t& result, uint64_t first, uint64_t copyData)
         {
            result = copyData;
         });
   }

   HeapMatrixState<uint64_t, TNumCharacters>& GetMatrix()
   {
      return this->mUnigrams;
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

   const IMatrixState<uint64_t, TNumCharacters, TNumCharacters>& GetMatrix() const
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

   void CopyFrom(const IMatrixState<uint64_t, TNumCharacters, TNumCharacters>& bigramData)
   {
      this->mBigrams.Zip(this->mBigrams, bigramData,
         [](uint64_t& result, uint64_t first, uint64_t copyData)
         {
            result = copyData;
         });
   }

   HeapMatrixState<uint64_t, TNumCharacters, TNumCharacters>& GetMatrix()
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
   GetMatrix() const
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

   void CopyFrom(
      const IMatrixState<uint64_t, TNumCharacters, TNumCharacters, TNumCharacters>& trigramData)
   {
      this->mTrigrams.Zip(this->mTrigrams, trigramData,
         [](uint64_t& result, uint64_t first, uint64_t copyData)
         {
            result = copyData;
         });
   }

   HeapMatrixState<uint64_t, TNumCharacters, TNumCharacters, TNumCharacters>& GetMatrix()
   {
      return this->mTrigrams;
   }

   void SetTrigramFrequency(int charIdx1, int charIdx2, int charIdx3, uint64_t frequency)
   {
      this->mTrigrams(charIdx1, charIdx2, charIdx3) = frequency;
   }

private:
};

/**
 * Stores the skipgram frequencies
 * The skipgrams are as a vector of bigrams where each the first index represents the
 * skipgram order, and the bigram frequency following represents the frequnecy of that bigram
 * such that skipgrams[0] -> ds[0] xx ds[3] where the number of characters between is equal to i
 * where we are using skipgrams[i - 2]
 *
 * Example: skipgrams[0] = 2 characters between
 *          skipgrams[3] = 5 characters between
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

   constexpr static int GetNumSkipgrams()
   {
      return TNumSkipgrams;
   }

   const IMatrixState<uint64_t, TNumSkipgrams, TNumCharacters, TNumCharacters>&
   GetMatrix() const
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

   void CopyFrom(
      const IMatrixState<uint64_t, TNumSkipgrams, TNumCharacters, TNumCharacters>& trigramData)
   {
      this->mSkipgrams.Zip(this->mSkipgrams, trigramData,
         [](uint64_t& result, uint64_t first, uint64_t copyData)
         {
            result = copyData;
         });
   }

   HeapMatrixState<uint64_t, TNumSkipgrams, TNumCharacters, TNumCharacters>& GetMatrix()
   {
      return this->mSkipgrams;
   }

   void SetSkipgramFrequency(int skipgramLength, int charIdx1, int charIdx2, uint64_t frequency)
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
   constexpr static int GetNumCharacters()
   {
      return TNumCharacters;
   }

   /**
    * @brief      The number of skipgrams supported.
    * @return     The number skipgrams.
    */
   constexpr static int GetNumSkipgrams()
   {
      return TNumSkipgrams;
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

   using unigrams_type = Unigrams<TNumCharacters>;
   using bigrams_type = Bigrams<TNumCharacters>;
   using trigrams_type = Trigrams<TNumCharacters>;
   using skipgrams_type = Skipgrams<TNumSkipgrams, TNumCharacters>;

private:
};

/**
 * @brief      Class instance responsible for loading text into a dataset in a memory efficient way
 * and flexible way. 
 *
 * @tparam     TDataset     The type of the dataset we are loading into.
 * @tparam     TCharMapper  The function which processes each character in the corpus and returns
 *                          the corresponding dataset character.
 */
template <typename TDataset, typename TCharMapper> class DatasetLoader
{
public:
   /**
    * @brief      Standard constructor.
    * @param      dataset  The dataset
    */
   DatasetLoader(TDataset& dataset)
   {
      // Reset is not called, but we expect the dataset to be cleared before loading the dataset.
   }

   /**
    * @brief      There is some state that must be reset if the dataset is going to be reloaded. Call
    * this to clear the existing data and reset that state.
    */
   void Reset()
   {
      mLeadingDataIdx = 0;
      mDataset.Clear();
   }

   /**
    * @brief      Loads the file in a chunked manner before processing and loading characters into
    * the dataset.
    *
    * @param      dataset       The dataset
    * @param[in]  maxChunkSize  The maximum chunk size
    * @param[in]  fileName      The file name
    */
   void LoadDataSetFromFileChunked(TDataset& dataset, int maxChunkSize, const std::string& fileName)
   {
   }

   /**
    * @brief      Processes each character in the chunk and appends the data to the corpus dataset.
    * Note, each character needs to be preprocessed into N dataset indices.
    *
    * @param      dataset   The dataset
    * @param[in]  chunk     The chunk
    */
   void LoadFromChunk(const std::string& chunk)
   {
      // Ensure we've loaded the entire leading pipeline of data.
      int chunkIndex = 0;
      while (!mLeadingDataIdx < TDataset::GetNumSkipgrams() && chunkIndex < chunk.length())
      {
         int dbIdx = mCharMapper(chunk[chunkIndex]);

         if (dbIdx >= 0)
         {
            mSkipgramsLeadingData[mLeadingDataIdx] = dbIdx;
            mLeadingDataIdx++;
         }

         chunkIndex++;
      }

      // Leave early if we don't have enough points in our pipeline.
      if (mLeadingDataIdx < TDataset::GetNumSkipgrams())
      {
         return;
      }

      auto& unigramData = mDataset.GetUnigrams().GetMatrix();
      auto& bigramData = mDataset.GetBigrams().GetMatrix();
      auto& trigramData = mDataset.GetTrigrams().GetMatrix();
      auto& skipgramData = mDataset.GetSkipgrams().GetMatrix();

      //tODO
      for (int i = 0; i < mLeadingDataIdx; i++)
      {
         int dbIdx = mCharMapper(mSkipgramsLeadingData[i]);

         unigramData.GetRef(dbIdx)++;
      }

      // Now that we have enough data, handle our leading character points.
      charFreq[charToIndex[filteredString[0]]]++;
      charFreq[charToIndex[filteredString[1]]]++;
      charFreq[charToIndex[filteredString[2]]]++;
      charFreq[charToIndex[filteredString[3]]]++;
      charFreq[charToIndex[filteredString[4]]]++;
      charFreq[charToIndex[filteredString[5]]]++;
      bigramFreq[GET_BG(0, 1)]++;
      bigramFreq[GET_BG(1, 2)]++;
      bigramFreq[GET_BG(2, 3)]++;
      bigramFreq[GET_BG(3, 4)]++;
      bigramFreq[GET_BG(4, 5)]++;

      trigramFreq[TAIDX(0, 1, 2)]++;
      trigramFreq[TAIDX(1, 2, 3)]++;
      trigramFreq[TAIDX(2, 3, 4)]++;
      trigramFreq[TAIDX(3, 4, 5)]++;

      // SkipGrams of [2]. (meaning two characters between).
      skipGramFreq[SKIDX(0, 0, 3)]++;
      skipGramFreq[SKIDX(0, 1, 4)]++;
      skipGramFreq[SKIDX(0, 2, 5)]++;

      // SkipGrams of [3]
      skipGramFreq[SKIDX(1, 0, 4)]++;
      skipGramFreq[SKIDX(1, 1, 5)]++;

      // SkipGrams of [4]
      skipGramFreq[SKIDX(2, 0, 5)]++;

      // The dataset is big enough to just skip the first couple characters, no problem.
      for (unsigned int i = 6; i < filteredString.size(); i++)
      {
         if (*canceled)
         {
            return -1;
         }

         charFreq[charToIndex[filteredString[i]]]++;

         // Deal with bigrams.
         bigramFreq[GET_BG(i - 1, i)]++;

         // Deal with trigrams.
         trigramFreq[TAIDX(i - 2, i - 1, i)]++;

         // Deal with skipgrams.
         skipGramFreq[SKIDX(0, i - 3, i)]++;
         skipGramFreq[SKIDX(1, i - 4, i)]++;
         skipGramFreq[SKIDX(2, i - 5, i)]++;

         *progress = (((double)i + 1) / (double)filteredString.size());
      }

      delete[] toLowerMap;

      return filteredString.size();
   }

private:
   TDataset& mDataset;
   std::decay_t<TCharMapper> mCharMapper;

   /**
    * @return      Returns the number of leading datapoints we need to add a chunk to the dataset.
    */
   constexpr int GetNumRequiredLeadingChars()
   {
      // We subtract 2 since we aren't using unigrams or bigrams when computing skipgrams.
      return TDataset::GetNumSkipgrams - 2;
   }
   int mLeadingDataIdx = 0;
   std::array<int, TDataset::GetNumSkipgrams()> mSkipgramsLeadingData;
};
}   // namespace kc