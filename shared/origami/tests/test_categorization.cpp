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
  REQUIRE(origami::classify_mn(1025) == origami::mn_range_t::large);
  REQUIRE(origami::classify_mn(4096) == origami::mn_range_t::large);
  REQUIRE(origami::classify_mn(4097) == origami::mn_range_t::xlarge);
}

TEST_CASE("Categorization: classify_k boundaries", "[categorization]") {
  REQUIRE(origami::classify_k(1) == origami::k_range_t::tiny);
  REQUIRE(origami::classify_k(128) == origami::k_range_t::tiny);
  REQUIRE(origami::classify_k(129) == origami::k_range_t::small);
  REQUIRE(origami::classify_k(512) == origami::k_range_t::small);
  REQUIRE(origami::classify_k(513) == origami::k_range_t::medium);
  REQUIRE(origami::classify_k(2048) == origami::k_range_t::medium);
  REQUIRE(origami::classify_k(2049) == origami::k_range_t::large);
  REQUIRE(origami::classify_k(8192) == origami::k_range_t::large);
  REQUIRE(origami::classify_k(8193) == origami::k_range_t::xlarge);
}

TEST_CASE("Categorization: id uniqueness and round-trip", "[categorization]") {
  std::set<std::size_t> ids;
  for (std::size_t i = 0; i < origami::NUM_GEMM_CATEGORIES; ++i) {
    auto cat = origami::category_from_id(i);
    REQUIRE(cat.id() == i);
    ids.insert(i);
  }
  REQUIRE(ids.size() == origami::NUM_GEMM_CATEGORIES);
}

TEST_CASE("Categorization: category_from_id out-of-range", "[categorization]") {
  REQUIRE_THROWS_AS(origami::category_from_id(125), std::out_of_range);
}

TEST_CASE("Categorization: corners", "[categorization]") {
  REQUIRE(origami::categorize_mnk(1, 1, 1).id() == 0);
  REQUIRE(origami::categorize_mnk(99999, 99999, 99999).id() == 124);
}

TEST_CASE("Categorization: full space coverage", "[categorization]") {
  std::vector<std::size_t> dims = {1, 64, 65, 256, 257, 1024, 1025, 4096, 4097, 16384};
  std::vector<std::size_t> k_dims = {1, 128, 129, 512, 513, 2048, 2049, 8192, 8193, 32768};

  for (auto m : dims)
    for (auto n : dims)
      for (auto k : k_dims) {
        auto cat = origami::categorize_mnk(m, n, k);
        REQUIRE(cat.id() < origami::NUM_GEMM_CATEGORIES);
        REQUIRE(m >= cat.m_lower());
        REQUIRE(n >= cat.n_lower());
        REQUIRE(k >= cat.k_lower());
        if (cat.m_upper() != SIZE_MAX) REQUIRE(m <= cat.m_upper());
        if (cat.n_upper() != SIZE_MAX) REQUIRE(n <= cat.n_upper());
        if (cat.k_upper() != SIZE_MAX) REQUIRE(k <= cat.k_upper());
      }
}

TEST_CASE("Categorization: K ranges match GridBased grid points", "[categorization]") {
  std::vector<std::size_t> grid_ks = {1, 48, 128, 192, 256, 512, 768, 1024,
                                       1536, 2048, 4096, 5120, 8192, 16384, 32768};
  std::vector<int> expected = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4};
  for (size_t i = 0; i < grid_ks.size(); ++i) {
    REQUIRE(static_cast<int>(origami::classify_k(grid_ks[i])) == expected[i]);
  }
}

TEST_CASE("Categorization: layout and dtype do NOT affect id", "[categorization]") {
  origami::problem_t p;
  p.size = {1024, 2048, 4096};
  p.a_transpose = origami::transpose_t::T;
  p.b_transpose = origami::transpose_t::N;
  p.mi_dtype = origami::data_type_t::BFloat16;
  auto id1 = origami::categorize(p).id();

  p.a_transpose = origami::transpose_t::N;
  p.b_transpose = origami::transpose_t::T;
  p.mi_dtype = origami::data_type_t::Float;
  auto id2 = origami::categorize(p).id();

  REQUIRE(id1 == id2);
}

// ========================================================================
// Training sample generation
// ========================================================================

TEST_CASE("Categorization: generate_training_samples count", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 512, 1024);
  auto samples = cat.generate_training_samples(4);
  REQUIRE(samples.size() == 4 * 4 * 4);
}

TEST_CASE("Categorization: samples are within category bounds", "[categorization]") {
  for (std::size_t id = 0; id < origami::NUM_GEMM_CATEGORIES; ++id) {
    auto cat = origami::category_from_id(id);
    auto samples = cat.generate_training_samples(3, 131072, 32768);

    for (const auto& s : samples) {
      REQUIRE(s.m >= cat.m_lower());
      REQUIRE(s.n >= cat.n_lower());
      REQUIRE(s.k >= cat.k_lower());
      auto m_ub = cat.m_upper() == SIZE_MAX ? static_cast<std::size_t>(131072) : cat.m_upper();
      auto n_ub = cat.n_upper() == SIZE_MAX ? static_cast<std::size_t>(131072) : cat.n_upper();
      auto k_ub = cat.k_upper() == SIZE_MAX ? static_cast<std::size_t>(32768) : cat.k_upper();
      REQUIRE(s.m <= m_ub);
      REQUIRE(s.n <= n_ub);
      REQUIRE(s.k <= k_ub);
    }
  }
}

TEST_CASE("Categorization: samples are log-spaced", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 512, 1024);
  auto samples = cat.generate_training_samples(4);

  double log_first_m = std::log2(static_cast<double>(samples.front().m));
  double log_last_m  = std::log2(static_cast<double>(samples[3 * 16].m));
  REQUIRE(log_last_m > log_first_m);
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

TEST_CASE("Categorization: ML features distinguish categories", "[categorization]") {
  auto f_small = origami::compute_ml_features(64, 64, 64);
  auto f_large = origami::compute_ml_features(4096, 4096, 4096);

  REQUIRE(f_large.log2_m > f_small.log2_m);
  REQUIRE(f_large.log2_mn_tiles > f_small.log2_mn_tiles);
  REQUIRE(f_large.arithmetic_intensity > f_small.arithmetic_intensity);
}

TEST_CASE("Categorization: ML features aspect ratio", "[categorization]") {
  auto f_tall = origami::compute_ml_features(8192, 64, 1024);
  auto f_wide = origami::compute_ml_features(64, 8192, 1024);
  auto f_sq   = origami::compute_ml_features(1024, 1024, 1024);

  REQUIRE(f_tall.mn_aspect_ratio > 0.0);
  REQUIRE(f_wide.mn_aspect_ratio < 0.0);
  REQUIRE(f_sq.mn_aspect_ratio == Approx(0.0));
}

TEST_CASE("Categorization: ML features k_mn_ratio", "[categorization]") {
  auto f_deep    = origami::compute_ml_features(256, 256, 32768);
  auto f_shallow = origami::compute_ml_features(256, 256, 64);

  REQUIRE(f_deep.k_mn_ratio > f_shallow.k_mn_ratio);
}

TEST_CASE("Categorization: AI formula", "[categorization]") {
  double m = 1024, n = 1024, k = 1024, bpe = 2.0;
  double expected = 2.0 * m * n * k / ((m * k + k * n + m * n) * bpe);
  REQUIRE(origami::compute_arithmetic_intensity(m, n, k, bpe) == Approx(expected));
}
