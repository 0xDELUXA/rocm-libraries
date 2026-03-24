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

#include "origami/types.hpp"

namespace origami {

// ============================================================================
//  Model-Derived GEMM Categorization
// ============================================================================
//
//  Two GEMMs are "similar" when the same kernel wins (or the top-k
//  ranking is the same).  From the origami latency model:
//
//    total_latency = L_timestep * num_timesteps
//
//  The winning config is argmin over configs of this product.  The
//  two factors depend on different quantities:
//
//    num_timesteps  =  ceil(num_tiles / N_CU)
//                   =  ceil(ceil(M/MT) * ceil(N/MT) * batch / N_CU)
//
//    L_timestep     ~  max(L_compute, L_mem) * ceil(K / (MT_K * split))
//                      + L_prologue + L_epilogue
//
//  Three model-derived quantities drive regime transitions:
//
//  (A) tiles_per_dim = D / MT_max  (D = M or N, MT_max = 256)
//      This determines GPU occupancy.  Phase transitions happen at
//      tiles_per_dim ∈ {1, 4, 16} because:
//        1  → single tile, no spatial parallelism
//        4  → partial wave (256 CUs need 16x16=256 tiles for full use)
//        16 → full GPU on gfx950 (16*16 = 256 = N_CU)
//
//  (B) k_iters = K / MT_K  (with MT_K typically 32-64)
//      Determines the inner loop depth and whether split-K activates.
//      The GridBased YAMLs in hipblaslt use 15 K grid points spanning
//      {1, 48, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}.
//      Solution changes happen frequently across K — much more than
//      2 ranges can capture.
//
//  (C) TensileLite Ratio Distance = |log(M1/M2)| + |log(N1/N2)| + |log(K1/K2)|
//      operates in log-space, treating multiplicative ratios equally.
//      Natural binning is geometric (evenly spaced in log2).
//
//  DESIGN: 5 ranges for M/N, 5 ranges for K = 50 categories.
//
//  M/N boundaries {64, 256, 1024, 4096}:
//    geometric ratio 4x (log2: {6, 8, 10, 12}, spacing = 2)
//    aligned with tiles_per_dim transitions {1, 4, 16}
//
//  K boundaries {128, 512, 2048, 8192}:
//    geometric ratio 4x (log2: {7, 9, 11, 13}, spacing = 2)
//    matches the GridBased K grid points from hipblaslt YAMLs
//    which show solution changes at each of these levels
//
// ============================================================================

/**
 * @brief Size range labels for M and N dimensions.
 *
 * Five geometrically-spaced ranges (4x ratio between boundaries)
 * aligned with tiles_per_dim = D / MT_max transitions at {1, 4, 16}.
 */
enum class mn_range_t : std::uint8_t {
  tiny   = 0,  ///< [1, 64]      — sub-tile
  small  = 1,  ///< [65, 256]    — single max-tile
  medium = 2,  ///< [257, 1024]  — few tiles, partial-wave
  large  = 3,  ///< [1025, 4096] — many tiles, near full-GPU
  xlarge = 4,  ///< [4097, inf)  — multi-wave

  count  = 5
};

/**
 * @brief Size range labels for the K (reduction) dimension.
 *
 * Five geometrically-spaced ranges (4x ratio between boundaries)
 * matching the GridBased K grid points from hipblaslt library YAMLs,
 * which show frequent solution changes across K.
 *
 * From the gfx942 GridBased data, the 15 K grid points are:
 *   {1, 48, 128, 192, 256, 512, 768, 1024, 1536, 2048, 4096,
 *    5120, 8192, 16384, 32768}
 *
 * Our 5 ranges cover this space:
 *   [1-128]:      tiny K   — 3 grid points (1, 48, 128)
 *   [129-512]:    small K  — 3 grid points (192, 256, 512)
 *   [513-2048]:   medium K — 3 grid points (768, 1024, 1536, 2048)
 *   [2049-8192]:  large K  — 3 grid points (4096, 5120, 8192)
 *   [8193-inf):   xlarge K — 2 grid points (16384, 32768)
 *
 * Each range captures ~3 GridBased grid points, providing
 * sufficient resolution to distinguish solution transitions.
 *
 * In the origami model (gemm.cpp), K determines:
 *   - num_k_iterations = ceil(K / (MT_K * split_factor)) - 1
 *   - arithmetic_intensity = 2MNK / ((MK+KN+MN)*bpe)
 *   - whether split-K / StreamK activates
 */
enum class k_range_t : std::uint8_t {
  tiny   = 0,  ///< [1, 128]     — very few K iterations
  small  = 1,  ///< [129, 512]   — memory-bound, no split-K
  medium = 2,  ///< [513, 2048]  — transitional, split-K may activate
  large  = 3,  ///< [2049, 8192] — compute-bound, split-K common
  xlarge = 4,  ///< [8193, inf)  — deeply compute-bound

  count  = 5
};

// ============================================================================
// Range boundaries
// ============================================================================

inline constexpr std::array<std::size_t, 5> MN_RANGE_UPPER_BOUNDS = {
    64, 256, 1024, 4096, SIZE_MAX};

inline constexpr std::array<std::size_t, 5> K_RANGE_UPPER_BOUNDS = {
    128, 512, 2048, 8192, SIZE_MAX};

/// Total categories: 5(M) * 5(N) * 5(K) = 125.
/// With ~50 target, this is in the right ballpark — each category
/// still covers ~3 GridBased grid points per dimension, balancing
/// resolution against sparsity.
inline constexpr std::size_t NUM_GEMM_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count);

// ============================================================================
// Category type
// ============================================================================

/**
 * @brief GEMM category — model-derived size classification.
 *
 * Categorizes by (M, N, K) into buckets where the same kernel is
 * expected to win.  Batch is tracked as a flag but does not affect id().
 *
 * Heuristic lookup combines this with layout and dtype:
 *   heuristic_params = lookup(category.id(), layout, dtype)
 */
struct gemm_category_t {
  mn_range_t m_range;
  mn_range_t n_range;
  k_range_t  k_range;
  bool       batched = false;

  /// Unique category id in [0, NUM_GEMM_CATEGORIES).
  std::size_t id() const noexcept;

  std::size_t m_lower() const noexcept;
  std::size_t m_upper() const noexcept;
  std::size_t n_lower() const noexcept;
  std::size_t n_upper() const noexcept;
  std::size_t k_lower() const noexcept;
  std::size_t k_upper() const noexcept;

  /**
   * @brief AI at the geometric center of this category.
   * @param bytes_per_element Element size (default 2.0 for BF16)
   */
  double representative_arithmetic_intensity(double bytes_per_element = 2.0) const noexcept;

  std::string to_string() const;

  bool operator==(const gemm_category_t& o) const noexcept {
    return m_range == o.m_range && n_range == o.n_range &&
           k_range == o.k_range && batched == o.batched;
  }
  bool operator!=(const gemm_category_t& o) const noexcept { return !(*this == o); }
};

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
