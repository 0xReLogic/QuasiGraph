#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace QuasiGraph {

/**
 * High-performance bitset implementation using SIMD operations
 * 
 * Features:
 * - AVX2 SIMD for fast set operations
 * - Hardware POPCNT for O(1) counting
 * - Cache-aligned memory layout
 * - 64-bit word-level operations
 */
class BitSet {
public:
    static constexpr size_t BITS_PER_WORD = 64;
    static constexpr size_t CACHE_LINE_SIZE = 64; // bytes

    BitSet() : size_(0), num_words_(0), words_(nullptr) {}
    
    explicit BitSet(size_t size) : size_(size) {
        num_words_ = (size + BITS_PER_WORD - 1) / BITS_PER_WORD;
        // Align to cache line for SIMD operations
        words_ = static_cast<uint64_t*>(aligned_alloc(CACHE_LINE_SIZE, 
                                        num_words_ * sizeof(uint64_t)));
        clear();
    }

    ~BitSet() {
        if (words_) {
            free(words_);
        }
    }

    // Copy constructor
    BitSet(const BitSet& other) : size_(other.size_), num_words_(other.num_words_) {
        if (other.words_) {
            words_ = static_cast<uint64_t*>(aligned_alloc(CACHE_LINE_SIZE,
                                            num_words_ * sizeof(uint64_t)));
            std::memcpy(words_, other.words_, num_words_ * sizeof(uint64_t));
        } else {
            words_ = nullptr;
        }
    }

    // Move constructor
    BitSet(BitSet&& other) noexcept 
        : size_(other.size_), num_words_(other.num_words_), words_(other.words_) {
        other.words_ = nullptr;
        other.size_ = 0;
        other.num_words_ = 0;
    }

    BitSet& operator=(const BitSet& other) {
        if (this != &other) {
            if (words_) free(words_);
            size_ = other.size_;
            num_words_ = other.num_words_;
            if (other.words_) {
                words_ = static_cast<uint64_t*>(aligned_alloc(CACHE_LINE_SIZE,
                                                num_words_ * sizeof(uint64_t)));
                std::memcpy(words_, other.words_, num_words_ * sizeof(uint64_t));
            } else {
                words_ = nullptr;
            }
        }
        return *this;
    }

    BitSet& operator=(BitSet&& other) noexcept {
        if (this != &other) {
            if (words_) free(words_);
            size_ = other.size_;
            num_words_ = other.num_words_;
            words_ = other.words_;
            other.words_ = nullptr;
            other.size_ = 0;
            other.num_words_ = 0;
        }
        return *this;
    }

    // Set bit at position
    inline void set(size_t pos) {
        if (pos >= size_) return;
        words_[pos / BITS_PER_WORD] |= (1ULL << (pos % BITS_PER_WORD));
    }

    // Clear bit at position
    inline void reset(size_t pos) {
        if (pos >= size_) return;
        words_[pos / BITS_PER_WORD] &= ~(1ULL << (pos % BITS_PER_WORD));
    }

    // Test bit at position
    inline bool test(size_t pos) const {
        if (pos >= size_) return false;
        return (words_[pos / BITS_PER_WORD] & (1ULL << (pos % BITS_PER_WORD))) != 0;
    }

    // Clear all bits
    inline void clear() {
        if (words_) {
            std::memset(words_, 0, num_words_ * sizeof(uint64_t));
        }
    }

    // Count set bits using hardware POPCNT
    inline size_t count() const {
        size_t total = 0;
        for (size_t i = 0; i < num_words_; ++i) {
            total += __builtin_popcountll(words_[i]);
        }
        return total;
    }

    // SIMD-accelerated intersection count (for dense graphs)
    inline size_t intersect_count(const BitSet& other) const {
        if (size_ != other.size_) return 0;
        
        size_t count = 0;
        
#ifdef __AVX2__
        // Process 4 words (256 bits) at a time with AVX2
        size_t simd_words = (num_words_ / 4) * 4;
        for (size_t i = 0; i < simd_words; i += 4) {
            __m256i a = _mm256_load_si256((__m256i*)(words_ + i));
            __m256i b = _mm256_load_si256((__m256i*)(other.words_ + i));
            __m256i intersection = _mm256_and_si256(a, b);
            
            // Count bits in each 64-bit word
            uint64_t* inter_words = (uint64_t*)&intersection;
            count += __builtin_popcountll(inter_words[0]);
            count += __builtin_popcountll(inter_words[1]);
            count += __builtin_popcountll(inter_words[2]);
            count += __builtin_popcountll(inter_words[3]);
        }
        
        // Handle remaining words
        for (size_t i = simd_words; i < num_words_; ++i) {
            count += __builtin_popcountll(words_[i] & other.words_[i]);
        }
#else
        // Fallback without SIMD
        for (size_t i = 0; i < num_words_; ++i) {
            count += __builtin_popcountll(words_[i] & other.words_[i]);
        }
#endif
        
        return count;
    }

    // SIMD-accelerated union
    inline void union_with(const BitSet& other) {
        if (size_ != other.size_) return;
        
#ifdef __AVX2__
        size_t simd_words = (num_words_ / 4) * 4;
        for (size_t i = 0; i < simd_words; i += 4) {
            __m256i a = _mm256_load_si256((__m256i*)(words_ + i));
            __m256i b = _mm256_load_si256((__m256i*)(other.words_ + i));
            __m256i result = _mm256_or_si256(a, b);
            _mm256_store_si256((__m256i*)(words_ + i), result);
        }
        
        for (size_t i = simd_words; i < num_words_; ++i) {
            words_[i] |= other.words_[i];
        }
#else
        for (size_t i = 0; i < num_words_; ++i) {
            words_[i] |= other.words_[i];
        }
#endif
    }

    // SIMD-accelerated intersection
    inline void intersect_with(const BitSet& other) {
        if (size_ != other.size_) return;
        
#ifdef __AVX2__
        size_t simd_words = (num_words_ / 4) * 4;
        for (size_t i = 0; i < simd_words; i += 4) {
            __m256i a = _mm256_load_si256((__m256i*)(words_ + i));
            __m256i b = _mm256_load_si256((__m256i*)(other.words_ + i));
            __m256i result = _mm256_and_si256(a, b);
            _mm256_store_si256((__m256i*)(words_ + i), result);
        }
        
        for (size_t i = simd_words; i < num_words_; ++i) {
            words_[i] &= other.words_[i];
        }
#else
        for (size_t i = 0; i < num_words_; ++i) {
            words_[i] &= other.words_[i];
        }
#endif
    }

    // Get all set bit positions (for iteration)
    inline std::vector<size_t> get_set_bits() const {
        std::vector<size_t> result;
        result.reserve(count());
        
        for (size_t i = 0; i < num_words_; ++i) {
            uint64_t word = words_[i];
            while (word) {
                size_t bit_pos = __builtin_ctzll(word); // Count trailing zeros
                result.push_back(i * BITS_PER_WORD + bit_pos);
                word &= word - 1; // Clear lowest set bit
            }
        }
        
        return result;
    }

    size_t size() const { return size_; }
    bool empty() const { return count() == 0; }

private:
    size_t size_;
    size_t num_words_;
    uint64_t* words_;
};

} // namespace QuasiGraph
