// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/categorization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace origami {

mn_range_t classify_mn(std::size_t dim) noexcept {
  for (std::size_t i = 0; i < MN_RANGE_UPPER_BOUNDS.size(); ++i) {
    if (dim <= MN_RANGE_UPPER_BOUNDS[i]) return static_cast<mn_range_t>(i);
  }
  return mn_range_t::xlarge;
}

k_range_t classify_k(std::size_t dim) noexcept {
  for (std::size_t i = 0; i < K_RANGE_UPPER_BOUNDS.size(); ++i) {
    if (dim <= K_RANGE_UPPER_BOUNDS[i]) return static_cast<k_range_t>(i);
  }
  return k_range_t::xlarge;
}

gemm_category_t categorize(const problem_t& problem) noexcept {
  return {classify_mn(problem.size.m),
          classify_mn(problem.size.n),
          classify_k(problem.size.k),
          problem.batch > 1};
}

gemm_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept {
  return {classify_mn(m), classify_mn(n), classify_k(k), false};
}

gemm_category_t category_from_id(std::size_t id) {
  if (id >= NUM_GEMM_CATEGORIES) {
    throw std::out_of_range("Category id " + std::to_string(id) +
                            " out of range [0, " +
                            std::to_string(NUM_GEMM_CATEGORIES) + ")");
  }
  const auto k_count = static_cast<std::size_t>(k_range_t::count);
  const auto n_count = static_cast<std::size_t>(mn_range_t::count);
  return {static_cast<mn_range_t>(id / (k_count * n_count)),
          static_cast<mn_range_t>((id / k_count) % n_count),
          static_cast<k_range_t>(id % k_count),
          false};
}

// ============================================================================
// gemm_category_t
// ============================================================================

std::size_t gemm_category_t::id() const noexcept {
  const auto k_count = static_cast<std::size_t>(k_range_t::count);
  const auto n_count = static_cast<std::size_t>(mn_range_t::count);
  return static_cast<std::size_t>(m_range) * n_count * k_count +
         static_cast<std::size_t>(n_range) * k_count +
         static_cast<std::size_t>(k_range);
}

static std::size_t mn_lower_bound(mn_range_t r) noexcept {
  auto idx = static_cast<std::size_t>(r);
  return idx == 0 ? 1 : MN_RANGE_UPPER_BOUNDS[idx - 1] + 1;
}

static std::size_t k_lower_bound(k_range_t r) noexcept {
  auto idx = static_cast<std::size_t>(r);
  return idx == 0 ? 1 : K_RANGE_UPPER_BOUNDS[idx - 1] + 1;
}

std::size_t gemm_category_t::m_lower() const noexcept { return mn_lower_bound(m_range); }
std::size_t gemm_category_t::m_upper() const noexcept { return MN_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(m_range)]; }
std::size_t gemm_category_t::n_lower() const noexcept { return mn_lower_bound(n_range); }
std::size_t gemm_category_t::n_upper() const noexcept { return MN_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(n_range)]; }
std::size_t gemm_category_t::k_lower() const noexcept { return k_lower_bound(k_range); }
std::size_t gemm_category_t::k_upper() const noexcept { return K_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(k_range)]; }

double gemm_category_t::representative_arithmetic_intensity(double bpe) const noexcept {
  constexpr double CAP_MN = 131072.0;
  constexpr double CAP_K  = 32768.0;
  auto gm = [](double lo, double hi, double cap) -> double {
    return std::sqrt(lo * ((hi == static_cast<double>(SIZE_MAX)) ? cap : hi));
  };
  return compute_arithmetic_intensity(
      gm(static_cast<double>(m_lower()), static_cast<double>(m_upper()), CAP_MN),
      gm(static_cast<double>(n_lower()), static_cast<double>(n_upper()), CAP_MN),
      gm(static_cast<double>(k_lower()), static_cast<double>(k_upper()), CAP_K),
      bpe);
}

// ============================================================================
// Training sample generation
// ============================================================================

static void add_if_in_range(std::vector<std::size_t>& out,
                            std::size_t val,
                            std::size_t lo, std::size_t hi) {
  if (val >= lo && val <= hi) out.push_back(val);
}

static std::vector<std::size_t> generate_dim_samples(
    std::size_t lo, std::size_t hi, std::size_t log_count,
    const std::vector<std::size_t>& tile_sizes) {

  std::vector<std::size_t> points;
  points.reserve(log_count * 8);

  double log_lo = std::log2(std::max(lo, static_cast<std::size_t>(1)));
  double log_hi = std::log2(std::max(hi, lo));

  // Layer 1: log-uniform base grid
  for (std::size_t i = 0; i < log_count; ++i) {
    double t = (log_count == 1) ? 0.5
             : static_cast<double>(i) / static_cast<double>(log_count - 1);
    auto val = static_cast<std::size_t>(std::round(std::exp2(log_lo + t * (log_hi - log_lo))));
    add_if_in_range(points, std::clamp(val, lo, hi), lo, hi);
  }

  // Layer 2: tile-boundary neighbors around log-spaced base points only.
  // For each base point, find the nearest tile boundary and include ±1.
  // This avoids O(hi/mt) iteration for large ranges.
  for (auto mt : tile_sizes) {
    for (std::size_t i = 0; i < log_count; ++i) {
      double t = (log_count == 1) ? 0.5
               : static_cast<double>(i) / static_cast<double>(log_count - 1);
      auto base = static_cast<std::size_t>(std::round(std::exp2(log_lo + t * (log_hi - log_lo))));

      // Nearest tile boundary below and above
      std::size_t below = (base / mt) * mt;
      std::size_t above = below + mt;

      for (auto boundary : {below, above}) {
        add_if_in_range(points, boundary, lo, hi);
        if (boundary > 0) add_if_in_range(points, boundary - 1, lo, hi);
        add_if_in_range(points, boundary + 1, lo, hi);
      }
    }
    // Also include small tile multiples at the range start
    for (std::size_t k = 1; k <= 4; ++k) {
      std::size_t boundary = mt * k;
      add_if_in_range(points, boundary, lo, hi);
      if (boundary > 0) add_if_in_range(points, boundary - 1, lo, hi);
      add_if_in_range(points, boundary + 1, lo, hi);
    }
  }

  // Layer 3: odd / prime / non-power-of-2
  constexpr std::size_t primes[] = {3, 7, 13, 17, 31, 37, 41, 47, 53, 61, 67, 71, 73, 79, 89, 97};
  for (auto p : primes) {
    add_if_in_range(points, p, lo, hi);
    for (auto mul : {10u, 100u, 1000u, 10000u}) {
      add_if_in_range(points, static_cast<std::size_t>(p) * mul, lo, hi);
    }
  }

  // Layer 4: cache-line alignment probes
  constexpr std::size_t cache_aligned[] = {64, 128, 192, 320, 384, 448, 576, 640, 768, 896};
  for (auto ca : cache_aligned) {
    add_if_in_range(points, ca, lo, hi);
    add_if_in_range(points, ca + 1, lo, hi);
  }

  // Deduplicate and sort
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());

  return points;
}

static std::vector<std::size_t> subsample(const std::vector<std::size_t>& v,
                                          std::size_t target) {
  if (v.size() <= target) return v;

  std::vector<std::size_t> out;
  out.reserve(target);

  // Always include first and last
  out.push_back(v.front());
  for (std::size_t i = 1; i < target - 1; ++i) {
    std::size_t idx = i * (v.size() - 1) / (target - 1);
    out.push_back(v[idx]);
  }
  out.push_back(v.back());

  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<dim3_t> gemm_category_t::generate_training_samples(
    std::size_t samples_per_dim, std::size_t max_samples,
    std::size_t cap_mn, std::size_t cap_k) const {

  std::size_t m_hi = (m_upper() == SIZE_MAX) ? cap_mn : m_upper();
  std::size_t n_hi = (n_upper() == SIZE_MAX) ? cap_mn : n_upper();
  std::size_t k_hi = (k_upper() == SIZE_MAX) ? cap_k  : k_upper();

  std::vector<std::size_t> mn_tiles = {32, 64, 96, 128, 160, 192, 208, 224, 256};
  std::vector<std::size_t> k_tiles  = {16, 32, 64, 128, 256, 512};

  auto m_points = generate_dim_samples(m_lower(), m_hi, samples_per_dim, mn_tiles);
  auto n_points = generate_dim_samples(n_lower(), n_hi, samples_per_dim, mn_tiles);
  auto k_points = generate_dim_samples(k_lower(), k_hi, samples_per_dim, k_tiles);

  // Cap dimensions so Cartesian product fits in max_samples.
  // Target: cbrt(max_samples) per dimension.
  std::size_t per_dim = std::max(
      static_cast<std::size_t>(std::cbrt(static_cast<double>(max_samples))),
      static_cast<std::size_t>(4));

  m_points = subsample(m_points, per_dim);
  n_points = subsample(n_points, per_dim);
  k_points = subsample(k_points, per_dim);

  std::vector<dim3_t> result;
  result.reserve(m_points.size() * n_points.size() * k_points.size());

  for (auto m : m_points)
    for (auto n : n_points)
      for (auto k : k_points)
        result.push_back({m, n, k});

  return result;
}

// ============================================================================
// ML Features
// ============================================================================

gemm_ml_features_t compute_ml_features(std::size_t m, std::size_t n, std::size_t k,
                                       double bytes_per_element) noexcept {
  double dm = std::max(static_cast<double>(m), 1.0);
  double dn = std::max(static_cast<double>(n), 1.0);
  double dk = std::max(static_cast<double>(k), 1.0);

  constexpr double MT_MAX = 256.0;

  return {std::log2(dm),
          std::log2(dn),
          std::log2(dk),
          compute_arithmetic_intensity(dm, dn, dk, bytes_per_element),
          std::log2(dm / dn),
          std::log2(dk / std::sqrt(dm * dn)),
          std::log2(dm * dn / (MT_MAX * MT_MAX))};
}

// ============================================================================
// Post-tuning analysis
// ============================================================================

category_analysis_t analyze_tuning_results(std::size_t category_id,
                                           const std::vector<tuning_result_t>& results) {
  category_analysis_t analysis{};
  analysis.category_id   = category_id;
  analysis.total_samples = results.size();

  if (results.empty()) {
    analysis.unique_winners       = 0;
    analysis.dominant_config_id   = 0;
    analysis.dominant_config_count = 0;
    analysis.purity               = 0.0;
    return analysis;
  }

  std::unordered_map<std::size_t, std::size_t> winner_counts;
  for (const auto& r : results) {
    winner_counts[r.best_config_id]++;
  }

  analysis.unique_winners = winner_counts.size();

  std::size_t max_count = 0;
  std::size_t max_id    = 0;
  for (const auto& [config_id, count] : winner_counts) {
    if (count > max_count) {
      max_count = count;
      max_id    = config_id;
    }
  }

  analysis.dominant_config_id    = max_id;
  analysis.dominant_config_count = max_count;
  analysis.purity = static_cast<double>(max_count) / static_cast<double>(results.size());

  return analysis;
}

// ============================================================================
// Strings
// ============================================================================

std::string gemm_category_t::to_string() const {
  auto fmt = [](std::size_t v) { return v == SIZE_MAX ? std::string("inf") : std::to_string(v); };
  auto pad = [](std::size_t v) { return (v < 10 ? "00" : v < 100 ? "0" : "") + std::to_string(v); };

  return "cat" + pad(id()) +
         "_M[" + std::to_string(m_lower()) + "-" + fmt(m_upper()) + "]" +
         "_N[" + std::to_string(n_lower()) + "-" + fmt(n_upper()) + "]" +
         "_K[" + std::to_string(k_lower()) + "-" + fmt(k_upper()) + "]" +
         "_" + (batched ? "batched" : "single");
}

double compute_arithmetic_intensity(double m, double n, double k, double bpe) noexcept {
  double bytes = (m * k + k * n + m * n) * bpe;
  return bytes > 0.0 ? 2.0 * m * n * k / bytes : 0.0;
}

}  // namespace origami
