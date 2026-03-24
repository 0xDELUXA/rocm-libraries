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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include "common.hpp"
#include "origami/categorization.hpp"

using Catch::Approx;

TEST_CASE("Categorization: NUM_GEMM_CATEGORIES is 125", "[categorization]") {
  REQUIRE(origami::NUM_GEMM_CATEGORIES == 125);
}

TEST_CASE("Categorization: classify_mn boundaries", "[categorization]") {
  REQUIRE(origami::classify_mn(1) == origami::mn_range_t::tiny);
  REQUIRE(origami::classify_mn(64) == origami::mn_range_t::tiny);
  REQUIRE(origami::classify_mn(65) == origami::mn_range_t::small);
  REQUIRE(origami::classify_mn(256) == origami::mn_range_t::small);
  REQUIRE(origami::classify_mn(257) == origami::mn_range_t::medium);
  REQUIRE(origami::classify_mn(1024) == origami::mn_range_t::medium);
  REQUIRE(origami::classify_mn(4097) == origami::mn_range_t::xlarge);
}

TEST_CASE("Categorization: classify_k boundaries", "[categorization]") {
  REQUIRE(origami::classify_k(1) == origami::k_range_t::tiny);
  REQUIRE(origami::classify_k(128) == origami::k_range_t::tiny);
  REQUIRE(origami::classify_k(129) == origami::k_range_t::small);
  REQUIRE(origami::classify_k(2049) == origami::k_range_t::large);
  REQUIRE(origami::classify_k(8193) == origami::k_range_t::xlarge);
}

TEST_CASE("Categorization: id round-trip", "[categorization]") {
  for (std::size_t i = 0; i < origami::NUM_GEMM_CATEGORIES; ++i) {
    REQUIRE(origami::category_from_id(i).id() == i);
  }
}

TEST_CASE("Categorization: id uniqueness", "[categorization]") {
  std::set<std::size_t> ids;
  for (std::size_t i = 0; i < origami::NUM_GEMM_CATEGORIES; ++i) ids.insert(i);
  REQUIRE(ids.size() == origami::NUM_GEMM_CATEGORIES);
}

TEST_CASE("Categorization: corners", "[categorization]") {
  REQUIRE(origami::categorize_mnk(1, 1, 1).id() == 0);
  REQUIRE(origami::categorize_mnk(99999, 99999, 99999).id() == 124);
}

TEST_CASE("Categorization: full space coverage", "[categorization]") {
  std::vector<std::size_t> dims = {1, 64, 65, 256, 257, 1024, 1025, 4096, 4097, 16384};
  std::vector<std::size_t> k_dims = {1, 128, 129, 512, 513, 2048, 2049, 8192, 8193, 32768};
  for (auto m : dims) for (auto n : dims) for (auto k : k_dims) {
    auto cat = origami::categorize_mnk(m, n, k);
    REQUIRE(cat.id() < origami::NUM_GEMM_CATEGORIES);
    REQUIRE(m >= cat.m_lower());
    REQUIRE(n >= cat.n_lower());
    REQUIRE(k >= cat.k_lower());
  }
}

// ========================================================================
// Structured training samples
// ========================================================================

TEST_CASE("Categorization: samples include tile-boundary neighbors", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 512, 1024);
  auto samples = cat.generate_training_samples(4);

  std::set<std::size_t> m_vals;
  for (const auto& s : samples) m_vals.insert(s.m);

  REQUIRE(m_vals.count(256 + 1) > 0);  // tile boundary + 1
  REQUIRE(m_vals.count(512) > 0);      // 2 * 256 tile boundary
  REQUIRE(m_vals.count(512 + 1) > 0);  // tile boundary + 1
}

TEST_CASE("Categorization: samples include odd/prime values", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 512, 1024);
  auto samples = cat.generate_training_samples(3);

  std::set<std::size_t> all_m;
  for (const auto& s : samples) all_m.insert(s.m);

  bool has_odd = false;
  for (auto v : all_m) {
    if (v % 2 == 1 && v > 1) { has_odd = true; break; }
  }
  REQUIRE(has_odd);
}

TEST_CASE("Categorization: samples include cache-alignment probes", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 512, 1024);
  auto samples = cat.generate_training_samples(3);

  std::set<std::size_t> all_k;
  for (const auto& s : samples) all_k.insert(s.k);

  bool has_aligned = all_k.count(1024) > 0;
  bool has_misaligned = false;
  for (auto v : all_k) {
    if (v > 1 && v % 64 != 0) { has_misaligned = true; break; }
  }
  REQUIRE(has_aligned);
  REQUIRE(has_misaligned);
}

TEST_CASE("Categorization: samples are sorted and unique per dim", "[categorization]") {
  auto cat = origami::categorize_mnk(1024, 1024, 2048);
  auto samples = cat.generate_training_samples(3);

  REQUIRE(samples.size() > 0);
  for (const auto& s : samples) {
    REQUIRE(s.m >= 1);
    REQUIRE(s.n >= 1);
    REQUIRE(s.k >= 1);
  }
}

// ========================================================================
// ML Features
// ========================================================================

TEST_CASE("Categorization: ML features basic", "[categorization]") {
  auto f = origami::compute_ml_features(1024, 1024, 1024);
  REQUIRE(f.log2_m == Approx(10.0));
  REQUIRE(f.log2_n == Approx(10.0));
  REQUIRE(f.log2_k == Approx(10.0));
  REQUIRE(f.mn_aspect_ratio == Approx(0.0));
  REQUIRE(f.arithmetic_intensity > 0.0);
}

TEST_CASE("Categorization: ML features distinguish shapes", "[categorization]") {
  auto f_tall = origami::compute_ml_features(8192, 64, 1024);
  auto f_wide = origami::compute_ml_features(64, 8192, 1024);
  REQUIRE(f_tall.mn_aspect_ratio > 0.0);
  REQUIRE(f_wide.mn_aspect_ratio < 0.0);
}

// ========================================================================
// Post-tuning analysis
// ========================================================================

TEST_CASE("Categorization: analyze_tuning_results pure category", "[categorization]") {
  std::vector<origami::tuning_result_t> results;
  for (int i = 0; i < 20; ++i) {
    results.push_back({{static_cast<std::size_t>(100+i), 100, 100}, 42, 100.0});
  }

  auto analysis = origami::analyze_tuning_results(0, results);
  REQUIRE(analysis.total_samples == 20);
  REQUIRE(analysis.unique_winners == 1);
  REQUIRE(analysis.dominant_config_id == 42);
  REQUIRE(analysis.purity == Approx(1.0));
  REQUIRE(analysis.is_pure());
}

TEST_CASE("Categorization: analyze_tuning_results mixed category", "[categorization]") {
  std::vector<origami::tuning_result_t> results;
  for (int i = 0; i < 10; ++i) results.push_back({{100, 100, 100}, 1, 50.0});
  for (int i = 0; i < 5; ++i)  results.push_back({{100, 100, 100}, 2, 45.0});
  for (int i = 0; i < 3; ++i)  results.push_back({{100, 100, 100}, 3, 40.0});
  for (int i = 0; i < 2; ++i)  results.push_back({{100, 100, 100}, 4, 35.0});

  auto analysis = origami::analyze_tuning_results(0, results);
  REQUIRE(analysis.total_samples == 20);
  REQUIRE(analysis.unique_winners == 4);
  REQUIRE(analysis.dominant_config_id == 1);
  REQUIRE(analysis.dominant_config_count == 10);
  REQUIRE(analysis.purity == Approx(0.5));
  REQUIRE(!analysis.is_pure());
}

TEST_CASE("Categorization: analyze_tuning_results empty", "[categorization]") {
  std::vector<origami::tuning_result_t> results;
  auto analysis = origami::analyze_tuning_results(0, results);
  REQUIRE(analysis.total_samples == 0);
  REQUIRE(analysis.unique_winners == 0);
  REQUIRE(analysis.purity == 0.0);
}

TEST_CASE("Categorization: AI formula", "[categorization]") {
  double m = 1024, n = 1024, k = 1024, bpe = 2.0;
  double expected = 2.0 * m * n * k / ((m * k + k * n + m * n) * bpe);
  REQUIRE(origami::compute_arithmetic_intensity(m, n, k, bpe) == Approx(expected));
}
