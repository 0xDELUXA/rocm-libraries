// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/categorization.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

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
  auto k_idx = id % k_count;
  auto n_idx = (id / k_count) % n_count;
  auto m_idx = id / (k_count * n_count);
  return {static_cast<mn_range_t>(m_idx),
          static_cast<mn_range_t>(n_idx),
          static_cast<k_range_t>(k_idx),
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
  constexpr double CAP = 32768.0;
  auto gm = [](double lo, double hi) -> double {
    return std::sqrt(lo * ((hi == static_cast<double>(SIZE_MAX)) ? CAP : hi));
  };
  return compute_arithmetic_intensity(
      gm(static_cast<double>(m_lower()), static_cast<double>(m_upper())),
      gm(static_cast<double>(n_lower()), static_cast<double>(n_upper())),
      gm(static_cast<double>(k_lower()), static_cast<double>(k_upper())),
      bpe);
}

// ============================================================================
// Training sample generation
// ============================================================================

static std::vector<std::size_t> log_uniform_samples(std::size_t lo, std::size_t hi,
                                                    std::size_t count, std::size_t cap) {
  if (hi == SIZE_MAX) hi = cap;
  lo = std::max(lo, static_cast<std::size_t>(1));
  hi = std::max(hi, lo);

  double log_lo = std::log2(static_cast<double>(lo));
  double log_hi = std::log2(static_cast<double>(hi));

  std::vector<std::size_t> samples;
  samples.reserve(count);

  if (count == 1) {
    samples.push_back(static_cast<std::size_t>(std::round(std::sqrt(lo * hi))));
    return samples;
  }

  for (std::size_t i = 0; i < count; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(count - 1);
    double log_val = log_lo + t * (log_hi - log_lo);
    auto val = static_cast<std::size_t>(std::round(std::exp2(log_val)));
    val = std::max(val, lo);
    val = std::min(val, hi);
    samples.push_back(val);
  }

  return samples;
}

std::vector<dim3_t> gemm_category_t::generate_training_samples(
    std::size_t samples_per_dim, std::size_t cap_mn, std::size_t cap_k) const {

  auto m_samples = log_uniform_samples(m_lower(), m_upper(), samples_per_dim, cap_mn);
  auto n_samples = log_uniform_samples(n_lower(), n_upper(), samples_per_dim, cap_mn);
  auto k_samples = log_uniform_samples(k_lower(), k_upper(), samples_per_dim, cap_k);

  std::vector<dim3_t> result;
  result.reserve(m_samples.size() * n_samples.size() * k_samples.size());

  for (auto m : m_samples)
    for (auto n : n_samples)
      for (auto k : k_samples)
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

  return {
      std::log2(dm),
      std::log2(dn),
      std::log2(dk),
      compute_arithmetic_intensity(dm, dn, dk, bytes_per_element),
      std::log2(dm / dn),
      std::log2(dk / std::sqrt(dm * dn)),
      std::log2(dm * dn / (MT_MAX * MT_MAX))};
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

// ============================================================================
// Arithmetic intensity
// ============================================================================

double compute_arithmetic_intensity(double m, double n, double k, double bpe) noexcept {
  double bytes = (m * k + k * n + m * n) * bpe;
  return bytes > 0.0 ? 2.0 * m * n * k / bytes : 0.0;
}

}  // namespace origami
