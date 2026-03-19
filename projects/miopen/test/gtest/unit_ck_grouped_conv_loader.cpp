// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/solver/ck_grouped_conv_lib_loader.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime.h>
#endif

namespace {

#if MIOPEN_BACKEND_HIP
std::string GetDeviceArch()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return {};
    return std::string(props.gcnArchName);
}
#endif

/// Build a minimal grouped-conv forward ProblemDescription suitable for
/// querying the CK loader.  Uses group=4, NHWC, FP16, small spatial dims.
miopen::conv::ProblemDescription MakeGroupedConvProblem()
{
    // Input:   N=1, C=16, H=8, W=8  (NHWC layout)
    // Weights: K=16, C/G=4, R=3, S=3 (NHWC layout, group=4 → C_per_group=4)
    // Conv:    pad=1, stride=1, dilation=1, group=4
    const miopen::TensorDescriptor in_desc(miopenHalf, miopenTensorNHWC, {1, 16, 8, 8});
    const miopen::TensorDescriptor wei_desc(miopenHalf, miopenTensorNHWC, {16, 4, 3, 3});

    const miopen::ConvolutionDescriptor conv_desc(
        /*pads=*/{1, 1},
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*trans_output_pads=*/{0, 0},
        /*group_count=*/4);

    const auto out_desc = conv_desc.GetForwardOutputTensor(in_desc, wei_desc, miopenHalf);

    return miopen::conv::ProblemDescription(
        in_desc, wei_desc, out_desc, conv_desc, miopen::conv::Direction::Forward);
}

} // namespace

// -- GPU tests (require a HIP device) -----------------------------------------

#if MIOPEN_BACKEND_HIP

TEST(GPU_CKGroupedConvLoader, LoaderLoadsForCurrentDevice)
{
    const auto device_name = GetDeviceArch();
    ASSERT_FALSE(device_name.empty()) << "Failed to query HIP device";

    const auto& loader = miopen::solver::CKGroupedConvLibLoader::Get(device_name);

    // The library may or may not be installed; if it loads, symbols must resolve.
    // We don't hard-fail on IsLoaded()==false because the .so might not be
    // present in all CI environments.
    if(loader.IsLoaded())
    {
        SUCCEED() << "Loader successfully loaded library for " << device_name;
    }
    else
    {
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;
    }
}

TEST(GPU_CKGroupedConvLoader, LoaderFillsValidKernels)
{
    const auto device_name = GetDeviceArch();
    ASSERT_FALSE(device_name.empty());

    const auto& loader = miopen::solver::CKGroupedConvLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem = MakeGroupedConvProblem();
    const auto kernels = loader.fwd_fill_valid_kernels(problem, miopenHalf, false);

    EXPECT_FALSE(kernels.empty())
        << "Expected at least one valid CK grouped conv kernel for " << device_name;
}

TEST(GPU_CKGroupedConvLoader, LoaderCachesPerDevice)
{
    const auto device_name = GetDeviceArch();
    ASSERT_FALSE(device_name.empty());

    const auto& loader1 = miopen::solver::CKGroupedConvLibLoader::Get(device_name);
    const auto& loader2 = miopen::solver::CKGroupedConvLibLoader::Get(device_name);

    EXPECT_EQ(&loader1, &loader2) << "Get() should return the same cached instance";
}

TEST(GPU_CKGroupedConvLoader, LoaderStripsDeviceSuffix)
{
    const auto device_name = GetDeviceArch();
    ASSERT_FALSE(device_name.empty());

    // Strip any existing suffix to get the base arch
    auto base_arch = device_name;
    auto colon_pos = base_arch.find(':');
    if(colon_pos != std::string::npos)
        base_arch = base_arch.substr(0, colon_pos);

    // Query with a synthesized suffix and with the bare base arch
    const auto suffixed   = base_arch + ":sramecc+:xnack-";
    const auto& loader1   = miopen::solver::CKGroupedConvLibLoader::Get(suffixed);
    const auto& loader2   = miopen::solver::CKGroupedConvLibLoader::Get(base_arch);

    EXPECT_EQ(&loader1, &loader2)
        << "Suffixed and bare device names should resolve to the same cached loader";
}

#endif // MIOPEN_BACKEND_HIP

// -- CPU tests (no GPU required) ----------------------------------------------

TEST(CPU_CKGroupedConvLoader, LoaderFailsGracefullyForUnknownDevice)
{
    const auto& loader = miopen::solver::CKGroupedConvLibLoader::Get("gfx_nonexistent");
    EXPECT_FALSE(loader.IsLoaded());
}

TEST(CPU_CKGroupedConvLoader, LoaderReturnsEmptyOnFailure)
{
    const auto& loader = miopen::solver::CKGroupedConvLibLoader::Get("gfx_bogus");
    ASSERT_FALSE(loader.IsLoaded());

    const auto problem = MakeGroupedConvProblem();

    // All wrappers should return safe defaults when the library is not loaded
    EXPECT_TRUE(loader.fwd_fill_valid_kernels(problem, miopenHalf, false).empty());
    EXPECT_FALSE(loader.fwd_is_applicable(problem, miopenHalf, false));
    EXPECT_FALSE(loader.fwd_is_args_supported(problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.fwd_get_workspace_size(problem, miopenHalf), 0u);

    EXPECT_TRUE(loader.bwd_fill_valid_kernels(problem, miopenHalf, false).empty());
    EXPECT_FALSE(loader.bwd_is_applicable(problem, miopenHalf, false));
    EXPECT_FALSE(loader.bwd_is_args_supported(problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.bwd_get_workspace_size(problem, miopenHalf), 0u);

    EXPECT_TRUE(loader.wrw_fill_valid_kernels(problem, miopenHalf, false).empty());
    EXPECT_FALSE(loader.wrw_is_applicable(problem, miopenHalf, false));
    EXPECT_FALSE(loader.wrw_is_args_supported(problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.wrw_get_workspace_size(problem, miopenHalf), 0u);
}
