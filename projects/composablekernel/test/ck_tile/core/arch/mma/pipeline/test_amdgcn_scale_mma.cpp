// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pipeline_tests_helper.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/scale/scale_mma_pipeline.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/utility/functional.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

// #include "get_wave_size_helper.hpp"
// #include "ck_tile/core/utility/type_traits.hpp"
// #include "ck_tile/host/hip_check_error.hpp"

// #include <hip/hip_runtime.h>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

using CompilerTargetGfx950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());

TEST(ScaleMMATrait, ScaleMfmaGfx950Specialization)
{
    // Test fp8 → fp32 scale MFMA for GFX950 (16x16x128)
    using TestScaleMfma16x16 = amdgcn_mma<fp8_t,
                                          fp8_t,
                                          fp32_t,
                                          16u,
                                          16u,
                                          128u,
                                          DefaultScaleMfmaCtrlFlags<fp8_t, fp8_t>,
                                          CompilerTargetGfx950,
                                          MmaOpFamily::SCALE>;

    static_assert(std::is_same_v<typename TestScaleMfma16x16::OpType, MfmaOp> &&
                      TestScaleMfma16x16::OpFamily == MmaOpFamily::SCALE,
                  "GFX950 scale 16x16x128 should have ScaleMFMAOp type");

    static_assert(is_mma_op_of_family_v<MmaOpFamily::SCALE, TestScaleMfma16x16>,
                  "GFX950 scale 16x16x128 should be detected as Scale");

    std::cout << "GFX950 scale MFMA specialization is correct" << std::endl;
}

TEST(ScaleMMATrait, MmaOpTraitsIntegration)
{
    // Create a scale MMA op (16x16x128 fp8 specialization)
    using TestScaleMma = amdgcn_mma<fp8_t,
                                    fp8_t,
                                    fp32_t,
                                    16u,
                                    16u,
                                    128u,
                                    DefaultScaleMfmaCtrlFlags<fp8_t, fp8_t>,
                                    CompilerTargetGfx950,
                                    MmaOpFamily::SCALE>;

    // Get its traits
    using TestTraits = MmaOpTraits<TestScaleMma>;

    // Verify trait detection
    static_assert(TestTraits::IsScale, "Scale MMA should be detected as scale");
    static_assert(TestTraits::IsSupported, "Scale MMA specialization should be supported");
    static_assert(TestTraits::IsMfma, "Scale MFMA should be detected as MFMA");
    static_assert(!TestTraits::IsWmma, "Scale MFMA should not be detected as WMMA");

    std::cout << "MmaOpTraits correctly integrates scale operations" << std::endl;
}

TEST(ScaleMMATrait, TestConceptRequirements)
{
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
    using TestScaleMma = amdgcn_mma<fp8_t,
                                    fp8_t,
                                    fp32_t,
                                    16u,
                                    16u,
                                    128u,
                                    DefaultScaleMfmaCtrlFlags<fp8_t, fp8_t>,
                                    CompilerTargetGfx950,
                                    MmaOpFamily::SCALE>;
    static_assert(MmaOpI<TestScaleMma>);
#else
    GTEST_SKIP() << "Not compiled with concepts. Skipping test.";
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
}

TEST(ScaleMMATrait, ScaleSelector)
{
    static_for<1, 33, 1>{}([](auto i) {
        using Selected = typename MmaDefaultSelector<fp8_t,
                                                     fp8_t,
                                                     fp32_t,
                                                     static_cast<std::uint32_t>(i),
                                                     static_cast<std::uint32_t>(i),
                                                     static_cast<std::uint32_t>(8 * i),
                                                     CompilerTargetGfx950,
                                                     MmaOpFamily::SCALE>::SelectedOp;

        static constexpr bool isValid = (i == 16) || (i == 32);
        if constexpr(isValid)
        {
            // Selector should pick a scale MFMA implementation
            static_assert(MmaOpTraits<Selected>::IsScale);
            static_assert(MmaOpTraits<Selected>::IsMfma);
            static_assert(MmaOpTraits<Selected>::IsSupported);
            static_assert((std::is_same<typename Selected::OpType, MfmaOp>::value));
        }
        else
        {
            // Selector should pick the unsupported pass through
            static_assert(!MmaOpTraits<Selected>::IsSupported);
        }
    });
}

template <typename AType,
          typename BType,
          typename CType,
          typename ScaleAType,
          typename ScaleBType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
__global__ void
test_scale_accum_over_k(void* a, void* b, void* c, void* out, void* scale_A, void* scale_B)
{
    using Pipeline = ScaleMmaPipeline<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>;

    using AVecType = typename Pipeline::AVecType;
    using BVecType = typename Pipeline::BVecType;
    using CVecType = typename Pipeline::CVecType;

    static constexpr std::uint32_t kIters = WaveTileK / Pipeline::MmaOp::kK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<CVecType*>(c);

    // Accumulate input AxB over WaveTileK/FragK iterations
    for(std::uint32_t i = 0; i < kIters; ++i)
    {
        result = Pipeline::exec(*reinterpret_cast<AVecType*>(a),
                                *reinterpret_cast<BVecType*>(b),
                                result,
                                *reinterpret_cast<ScaleAType*>(scale_A),
                                *reinterpret_cast<ScaleBType*>(scale_B));
    }

    *reinterpret_cast<CVecType*>(out) = result;
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F8_F8_F32_16x16x128_Real)
{
    using TestType = MmaPipelineTest<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>;
    TestType test;
    const auto should_skip = [](amdgcn_target_id currentArchId) {
        bool isSupportedWmma = false;
        bool isSupportedMfma = (currentArchId == amdgcn_target_id::GFX950);
        return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
    };
    const std::function<fp32_t(std::uint32_t, TestType::ScaleAType, TestType::ScaleBType)>
        validator =
            [](std::uint32_t fragK, TestType::ScaleAType scale_A, TestType::ScaleBType scale_B) {
                fp32_t actual_scale_A = std::powf(2.0f, scale_A - 127.0f);
                fp32_t actual_scale_B = std::powf(2.0f, scale_B - 127.0f);
                return static_cast<fp32_t>(fragK) * actual_scale_A * actual_scale_B;
            };
    const auto kernel = [](std::uint32_t waveSize,
                           void* a,
                           void* b,
                           void* c,
                           void* out,
                           void* scale_A,
                           void* scale_B) {
        test_scale_accum_over_k<TestType::AType,
                                TestType::BType,
                                TestType::CType,
                                TestType::ScaleAType,
                                TestType::ScaleBType,
                                TestType::WaveTileM,
                                TestType::WaveTileN,
                                TestType::WaveTileK>
            <<<1, waveSize>>>(a, b, c, out, scale_A, scale_B);
    };
    test.test_pipeline(should_skip, kernel, validator);
}
