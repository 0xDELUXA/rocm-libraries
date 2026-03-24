/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "origami/types.hpp"

namespace origami {

// ============================================================================
//  GEMM Categorization for ML Kernel Recommender Training
// ============================================================================
//
//  125 categories: 5(M) x 5(N) x 5(K), geometric 4x progression.
//
//  M/N boundaries: {64, 256, 1024, 4096}  — log2: {6, 8, 10, 12}
//  K boundaries:   {128, 512, 2048, 8192} — log2: {7, 9, 11, 13}
//
// ============================================================================

enum class mn_range_t : std::uint8_t {
  tiny   = 0,  ///< [1, 64]
  small  = 1,  ///< [65, 256]
  medium = 2,  ///< [257, 1024]
  large  = 3,  ///< [1025, 4096]
  xlarge = 4,  ///< [4097, inf)

  count  = 5
};

enum class k_range_t : std::uint8_t {
  tiny   = 0,  ///< [1, 128]
  small  = 1,  ///< [129, 512]
  medium = 2,  ///< [513, 2048]
  large  = 3,  ///< [2049, 8192]
  xlarge = 4,  ///< [8193, inf)

  count  = 5
};

inline constexpr std::array<std::size_t, 5> MN_RANGE_UPPER_BOUNDS = {
    64, 256, 1024, 4096, SIZE_MAX};

inline constexpr std::array<std::size_t, 5> K_RANGE_UPPER_BOUNDS = {
    128, 512, 2048, 8192, SIZE_MAX};

inline constexpr std::size_t NUM_GEMM_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count);

// ============================================================================
// Category type
// ============================================================================

struct gemm_category_t {
  mn_range_t m_range;
  mn_range_t n_range;
  k_range_t  k_range;
  bool       batched = false;

  std::size_t id() const noexcept;

  std::size_t m_lower() const noexcept;
  std::size_t m_upper() const noexcept;
  std::size_t n_lower() const noexcept;
  std::size_t n_upper() const noexcept;
  std::size_t k_lower() const noexcept;
  std::size_t k_upper() const noexcept;

  double representative_arithmetic_intensity(double bytes_per_element = 2.0) const noexcept;

  /**
   * @brief Generate structured training samples within this category.
   *
   * Produces points that exercise the decision boundaries in
   * origami's latency model.  Per dimension, generates:
   *
   *   1. LOG-SPACED GRID: samples_per_dim points uniform in
   *      log2-space (matching TensileLite Ratio distance)
   *
   *   2. TILE-BOUNDARY NEIGHBORS: around each log-spaced point,
   *      nearest k*MT and k*MT ± 1 for tile sizes {32..256}.
   *      work_utilization has discontinuities at tile multiples.
   *
   *   3. CACHE-ALIGNMENT PROBES: values where dim*bpe is/isn't
   *      a multiple of 128B.  gemm.cpp short-circuit tests this.
   *
   *   4. ODD / PRIME VALUES: stress edge padding and
   *      vectorization remainder paths.
   *
   * The final sample set is the Cartesian product of per-dimension
   * points, capped at max_samples.  If the Cartesian product
   * exceeds max_samples, dimensions are subsampled uniformly.
   *
   * @param samples_per_dim Log-spaced base points per dimension
   * @param max_samples Maximum total (M,N,K) triples to return
   * @param cap_mn Upper cap for unbounded M/N ranges
   * @param cap_k  Upper cap for unbounded K range
   * @return Vector of (M, N, K) triples
   */
  std::vector<dim3_t> generate_training_samples(std::size_t samples_per_dim = 8,
                                                std::size_t max_samples = 8000,
                                                std::size_t cap_mn = 131072,
                                                std::size_t cap_k = 32768) const;

  std::string to_string() const;

  bool operator==(const gemm_category_t& o) const noexcept {
    return m_range == o.m_range && n_range == o.n_range &&
           k_range == o.k_range && batched == o.batched;
  }
  bool operator!=(const gemm_category_t& o) const noexcept { return !(*this == o); }
};

// ============================================================================
// ML Feature Vector
// ============================================================================

/**
 * @brief ML features for kernel recommendation, derived from origami's model.
 */
struct gemm_ml_features_t {
  double log2_m;                 ///< log2(M) — TensileLite Ratio distance axis
  double log2_n;                 ///< log2(N)
  double log2_k;                 ///< log2(K)
  double arithmetic_intensity;   ///< 2MNK/((MK+KN+MN)*bpe) — roofline regime
  double mn_aspect_ratio;        ///< log2(M/N) — tall vs wide
  double k_mn_ratio;             ///< log2(K/sqrt(MN)) — depth vs output
  double log2_mn_tiles;          ///< log2(MN/256^2) — GPU occupancy
};

gemm_ml_features_t compute_ml_features(std::size_t m, std::size_t n, std::size_t k,
                                       double bytes_per_element = 2.0) noexcept;

// ============================================================================
// Post-tuning analysis types
// ============================================================================

/**
 * @brief Tuning result for a single (M, N, K) point.
 *
 * After running origami::rank_configs() (or actual benchmarking),
 * record which kernel won and its performance.
 */
struct tuning_result_t {
  dim3_t      size;           ///< (M, N, K)
  std::size_t best_config_id; ///< index of winning MT×MI config
  double      best_gflops;    ///< achieved performance
};

/**
 * @brief Post-tuning analysis for one category.
 *
 * After collecting tuning_result_t for all training samples in a
 * category, this structure answers:
 *
 *   1. DOMINANT KERNEL: which config wins most often?
 *      If one kernel wins >80% of samples, a simple heuristic
 *      suffices — no ML model needed for this category.
 *
 *   2. KERNEL COUNT: how many distinct kernels are competitive?
 *      If only 2-3, the ML model is a simple classifier.
 *      If 20+, the category may need splitting.
 *
 *   3. COVERAGE: what fraction of the category's sample space
 *      was actually tested? (for confidence estimation)
 *
 *   4. PURITY: if one kernel dominates, the category is "pure"
 *      and the ML model can be simple or omitted.
 */
struct category_analysis_t {
  std::size_t category_id;
  std::size_t total_samples;
  std::size_t unique_winners;  ///< number of distinct winning configs

  std::size_t dominant_config_id;    ///< most frequent winner
  std::size_t dominant_config_count; ///< how many samples it won
  double      purity;               ///< dominant_count / total_samples

  /// If purity > this threshold, no ML model needed — use the dominant kernel.
  static constexpr double PURE_THRESHOLD = 0.80;

  bool is_pure() const noexcept { return purity >= PURE_THRESHOLD; }
};

/**
 * @brief Analyze tuning results for a category.
 *
 * @param category_id The category that was tuned
 * @param results All tuning results within this category
 * @return category_analysis_t Analysis summary
 */
category_analysis_t analyze_tuning_results(std::size_t category_id,
                                           const std::vector<tuning_result_t>& results);

// ============================================================================
// Classification functions
// ============================================================================

mn_range_t classify_mn(std::size_t dim) noexcept;
k_range_t classify_k(std::size_t dim) noexcept;

gemm_category_t categorize(const problem_t& problem) noexcept;
gemm_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept;
gemm_category_t category_from_id(std::size_t id);

double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element = 2.0) noexcept;

}  // namespace origami
