// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>

#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace {

using miopen::conv::ProblemDescription;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGWrwPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    miopen::solver::conv::DeviceOpGWrw<DataType, ComputeType>>;

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
        using miopen::solver::ProblemInterpreter;

        G               = ProblemInterpreter::GetGroupCountG(problem);
        N               = ProblemInterpreter::GetBatchN(problem);
        K1              = ProblemInterpreter::GetOutputChannelK(problem);
        C1              = ProblemInterpreter::GetInputChannelC(problem);
        C               = C1 / G;
        K               = K1 / G;
        Hi              = ProblemInterpreter::GetInputHeightHi(problem);
        Wi              = ProblemInterpreter::GetInputWidthWi(problem);
        Ho              = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo              = ProblemInterpreter::GetOutputWidthWo(problem);
        Y               = ProblemInterpreter::GetFilterHeightY(problem);
        X               = ProblemInterpreter::GetFilterWidthX(problem);
        data_type       = ProblemInterpreter::GetOutputDataType(problem);
        alpha_beta_case = ProblemInterpreter::GetAlphaBetaCase(problem);
        input           = {G, N, C, Hi, Wi};
        output          = {G, N, K, Ho, Wo};
        weight          = {G, K, C, Y, X};

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
                    ConstData_t x,
                    Data_t dw,
                    ConstData_t dy,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             output,
                                             out_strides,
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
                    const miopen::ConvWrwTensors& tensors,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        return MakeArgPtr(conv_ptr, tensors.x, tensors.dw, tensors.dy, alpha, beta, split_k);
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
        if(use_tf32 &&
           miopen::solver::IsCKApplicable<DeviceOpGWrwPtrs<DataType, ck::tf32_t>, CKArgs>(problem))
        {
            return true;
        }
    }
    return miopen::solver::IsCKApplicable<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
}

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
        {
            return miopen::solver::FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType, ck::tf32_t>,
                                                       CKArgs>(problem);
        }
    }
    return miopen::solver::FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32 &&
           miopen::solver::IsCKArgsSupported<DeviceOpGWrwPtrs<DataType, ck::tf32_t>, CKArgs>(
               problem, kernel_id))
        {
            return true;
        }
    }
    return miopen::solver::IsCKArgsSupported<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem,
                                                                                 kernel_id);
}

template <typename DataType>
size_t GetWorkspaceSize(const ProblemDescription& problem)
{
    return miopen::solver::GetCKSplitkMaxWorkspaceSize<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// extern "C" WRW implementations
// ---------------------------------------------------------------------------

extern "C" {

CKKernelListHandle* ckgrpconv_wrw_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32)
{
    try
    {
        auto result = std::make_unique<CKKernelListHandle>();
        switch(data_type)
        {
        case miopenHalf: result->kernels = FillValidKernels<ck::half_t>(*problem, use_tf32); break;
        case miopenFloat: result->kernels = FillValidKernels<float>(*problem, use_tf32); break;
        case miopenInt8: result->kernels = FillValidKernels<int8_t>(*problem, use_tf32); break;
        case miopenBFloat16:
            result->kernels = FillValidKernels<ck::bhalf_t>(*problem, use_tf32);
            break;
        default: return nullptr;
        }
        return result.release();
    }
    catch(...)
    {
        return nullptr;
    }
}

bool ckgrpconv_wrw_is_applicable(const miopen::conv::ProblemDescription* problem,
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
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenDouble:
        default: break;
        }
        return false;
    }
    catch(...)
    {
        return false;
    }
}

bool ckgrpconv_wrw_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenDouble:
        default: break;
        }
        return false;
    }
    catch(...)
    {
        return false;
    }
}

size_t ckgrpconv_wrw_get_workspace_size(const miopen::conv::ProblemDescription* problem,
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
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenDouble:
        default: break;
        }
        return 0;
    }
    catch(...)
    {
        return 0;
    }
}

miopen::solver::ConvSolution*
ckgrpconv_wrw_get_solution(const miopen::ExecutionContext* ctx,
                           const miopen::conv::ProblemDescription* problem,
                           const char* kernel_id,
                           bool use_tf32)
{
    try
    {
        if(!ctx || !problem || !kernel_id)
            return nullptr;

        std::string kid(kernel_id);

        auto solution = miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return miopen::solver::InitInvokerFactoryWrwNCHW<2,
                                                                 false,
                                                                 DeviceOpGWrwPtrs<T, TCompute>,
                                                                 CKArgs,
                                                                 miopen::conv::WrWInvokeParams>(
                    *ctx, *problem, kid);
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return miopen::solver::InitInvokerFactoryNHWC<false,
                                                              DeviceOpGWrwPtrs<T, TCompute>,
                                                              CKArgs,
                                                              miopen::conv::WrWInvokeParams>(
                    *ctx, *problem, kid);
            },
            use_tf32);

        return new miopen::solver::ConvSolution(std::move(solution));
    }
    catch(...)
    {
        return nullptr;
    }
}

} // extern "C"
