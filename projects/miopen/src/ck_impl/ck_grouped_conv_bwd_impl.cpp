// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cassert>

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>

using miopen::conv::ProblemDescription;
using miopen::solver::ConvSolution;
using miopen::solver::FillValidKernelsIDs;
using miopen::solver::GetCKSplitkMaxWorkspaceSize;
using miopen::solver::InitInvokerFactoryBwdNCHW;
using miopen::solver::InitInvokerFactoryNHWC;
using miopen::solver::IsCKApplicable;
using miopen::solver::IsCKArgsSupported;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;
using miopen::solver::ProblemInterpreter;

using miopen::solver::conv::DeviceOpGBwdPtrs;

namespace {

// CKArgs — extracts convolution dimensions for CK argument construction.
//
// Each direction (fwd, bwd, wrw) has its own CKArgs with the same dimension
// members but direction-specific MakeArgPtr tensor ordering and IsSupportedBy
// logic.  BWD and WRW additionally handle NHWC layout strides and split-k
// workspace queries.  See ck_grouped_conv_fwd_impl.cpp for the FWD variant.
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        auto d = ExtractConvDims(problem);
        G = d.G; N = d.N; K1 = d.K1; C1 = d.C1; C = d.C; K = d.K;
        Hi = d.Hi; Wi = d.Wi; Ho = d.Ho; Wo = d.Wo; Y = d.Y; X = d.X;
        data_type       = ProblemInterpreter::GetOutputDataType(problem);
        alpha_beta_case = ProblemInterpreter::GetAlphaBetaCase(problem);

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        if(problem.IsLayoutNHWC())
        {
            auto copy_strides = [](const auto& src, auto& dst) {
                assert(dst.size() == (src.size() + 1));
                std::copy(src.begin(), src.end(), dst.begin() + 1);
            };
            copy_strides(problem.GetIn().GetStrides(), in_strides);
            copy_strides(problem.GetOut().GetStrides(), out_strides);
            copy_strides(problem.GetWeights().GetStrides(), wei_strides);

            // On a backward pass, problem.GetIn() means y(or out),
            // and problem.GetOut means x(or in)
            std::swap(in_strides, out_strides);

            in_strides[0]  = C;
            out_strides[0] = K;
            wei_strides[0] = K * wei_strides[1];
        }
        else
        {
            assert(problem.IsLayoutDefault());
            in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
            out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
            wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        }

        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    Data_t in,
                    ConstData_t w,
                    ConstData_t out,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             output,
                                             out_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             input,
                                             in_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const miopen::ConvDataTensors& tensors,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        return MakeArgPtr(conv_ptr, tensors.out, tensors.w, tensors.in, alpha, beta, split_k);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr        = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, 1);
        auto workspace_size = conv_ptr->GetWorkSpaceSize(arg_ptr.get());
        if(workspace_size != 0)
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &workspace_size);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr        = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);
        auto workspace_size = conv_ptr->GetWorkSpaceSize(arg_ptr.get());
        if(workspace_size != 0)
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &workspace_size);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    std::size_t GetCKSplitkWorkspaceSize(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);
        return conv_ptr->GetWorkSpaceSize(arg_ptr.get());
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    miopenDataType_t data_type;
    miopenAlphaBetaCase_t alpha_beta_case;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

template <typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem, bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32 && IsCKApplicable<DeviceOpGBwdPtrs<DataType, ck::tf32_t>, CKArgs>(problem))
            return true;
    }
    return IsCKApplicable<DeviceOpGBwdPtrs<DataType>, CKArgs>(problem);
}

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
        {
            return FillValidKernelsIDs<DeviceOpGBwdPtrs<DataType, ck::tf32_t>, CKArgs>(problem);
        }
    }
    return FillValidKernelsIDs<DeviceOpGBwdPtrs<DataType>, CKArgs>(problem);
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32 &&
           IsCKArgsSupported<DeviceOpGBwdPtrs<DataType, ck::tf32_t>, CKArgs>(problem, kernel_id))
        {
            return true;
        }
    }
    return IsCKArgsSupported<DeviceOpGBwdPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
size_t GetWorkspaceSize(const ProblemDescription& problem)
{
    return GetCKSplitkMaxWorkspaceSize<DeviceOpGBwdPtrs<DataType>, CKArgs>(problem);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// extern "C" BWD implementation
// ---------------------------------------------------------------------------

extern "C" CKKernelListHandle* ckgrpconv_bwd_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32)
{
    try
    {
        auto result = std::make_unique<CKKernelListHandle>();
        switch(data_type)
        {
        case miopenHalf: result->kernels = FillValidKernels<ck::half_t>(*problem, use_tf32); break;
        case miopenFloat: result->kernels = FillValidKernels<float>(*problem, use_tf32); break;
        case miopenBFloat16:
            result->kernels = FillValidKernels<ck::bhalf_t>(*problem, use_tf32);
            break;
        case miopenInt8: result->kernels = FillValidKernels<int8_t>(*problem, use_tf32); break;
        default: return nullptr;
        }
        return result.release();
    }
    catch(...)
    {
        return nullptr;
    }
}

extern "C" bool ckgrpconv_bwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                                            miopenDataType_t data_type,
                                            bool use_tf32)
{
    try
    {
        switch(data_type)
        {
        case miopenHalf: return CheckCKApplicability<ck::half_t>(*problem, use_tf32);
        case miopenFloat: return CheckCKApplicability<float>(*problem, use_tf32);
        case miopenInt8: return CheckCKApplicability<int8_t>(*problem, use_tf32);
        case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(*problem, use_tf32);
        default: return false;
        }
    }
    catch(...)
    {
        return false;
    }
}

extern "C" bool ckgrpconv_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                                const char* kernel_id,
                                                miopenDataType_t data_type,
                                                bool use_tf32)
{
    try
    {
        if(!kernel_id)
            return false;
        std::string kid(kernel_id);
        switch(data_type)
        {
        case miopenHalf: return CheckIsArgSupported<ck::half_t>(*problem, kid, use_tf32);
        case miopenFloat: return CheckIsArgSupported<float>(*problem, kid, use_tf32);
        case miopenInt8: return CheckIsArgSupported<int8_t>(*problem, kid, use_tf32);
        case miopenBFloat16: return CheckIsArgSupported<ck::bhalf_t>(*problem, kid, use_tf32);
        default: return false;
        }
    }
    catch(...)
    {
        return false;
    }
}

extern "C" size_t ckgrpconv_bwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                                   miopenDataType_t data_type)
{
    try
    {
        switch(data_type)
        {
        case miopenHalf: return GetWorkspaceSize<ck::half_t>(*problem);
        case miopenFloat: return GetWorkspaceSize<float>(*problem);
        case miopenInt8: return GetWorkspaceSize<int8_t>(*problem);
        case miopenBFloat16: return GetWorkspaceSize<ck::bhalf_t>(*problem);
        default: return 0;
        }
    }
    catch(...)
    {
        return 0;
    }
}

extern "C" miopen::solver::ConvSolution*
ckgrpconv_bwd_get_solution(const miopen::ExecutionContext* ctx,
                           const miopen::conv::ProblemDescription* problem,
                           const char* kernel_id,
                           bool use_tf32)
{
    try
    {
        if(!ctx || !problem || !kernel_id)
            return nullptr;
        std::string kid(kernel_id);

        auto solution = MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryBwdNCHW<2,
                                                 false,
                                                 DeviceOpGBwdPtrs<T, TCompute>,
                                                 CKArgs,
                                                 miopen::conv::DataInvokeParams>(
                    *ctx, *problem, kid);
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdPtrs<T, TCompute>,
                                              CKArgs,
                                              miopen::conv::DataInvokeParams>(*ctx, *problem, kid);
            },
            use_tf32);

        return new ConvSolution(std::move(solution));
    }
    catch(...)
    {
        return nullptr;
    }
}
