// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>

// ---------------------------------------------------------------------------
// CK type aliases for grouped convolution forward
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwd = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NHWGK,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGFwd<DataType, ComputeType>>;

// ---------------------------------------------------------------------------
// CKArgs — extracts convolution dimensions for CK argument construction.
//
// Each direction (fwd, bwd, wrw) has its own CKArgs with the same dimension
// members but direction-specific MakeArgPtr tensor ordering and IsSupportedBy
// logic.  FWD does not use split-k.  BWD/WRW additionally handle NHWC layout
// strides and split-k workspace queries.
// ---------------------------------------------------------------------------
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G;
        K  = K1 / G;
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        // strides from NHWGC to GNCHW layout
        in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
        out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
        wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        strides     = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                       ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation    = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                       ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding    = {ProblemInterpreter::GetInputLeftPadH(problem),
                       ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding    = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                       ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    Data_t out,
                    float alpha,
                    float beta) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {},
                                             out,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             output,
                                             out_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const miopen::ConvDataTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.in, tensors.w, tensors.out, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K1;
    int C1;
    int K;
    int C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
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

} // anonymous namespace

// ===========================================================================
// FWD direction functions
// ===========================================================================

using miopen::solver::FillValidKernelsIDs;
using miopen::solver::IsCKApplicable;
using miopen::solver::IsCKArgsSupported;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;
using miopen::solver::InitInvokerFactoryFwdNCHW;
using miopen::solver::InitInvokerFactoryNHWC;

extern "C" CKKernelListHandle* ckgrpconv_fwd_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem,
    miopenDataType_t data_type,
    bool use_tf32)
{
    try
    {
        auto result = std::make_unique<CKKernelListHandle>();
        switch(data_type)
        {
        case miopenHalf:
            result->kernels =
                FillValidKernelsIDs<DeviceOpGFwdPtrs<ck::half_t>, CKArgs>(*problem);
            break;
        case miopenFloat:
            if(use_tf32)
                result->kernels =
                    FillValidKernelsIDs<DeviceOpGFwdPtrs<float, ck::tf32_t>, CKArgs>(*problem);
            else
                result->kernels =
                    FillValidKernelsIDs<DeviceOpGFwdPtrs<float>, CKArgs>(*problem);
            break;
        case miopenBFloat16:
            result->kernels =
                FillValidKernelsIDs<DeviceOpGFwdPtrs<ck::bhalf_t>, CKArgs>(*problem);
            break;
        case miopenInt8:
            result->kernels =
                FillValidKernelsIDs<DeviceOpGFwdPtrs<int8_t>, CKArgs>(*problem);
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

extern "C" bool ckgrpconv_fwd_is_applicable(
    const miopen::conv::ProblemDescription* problem,
    miopenDataType_t data_type,
    bool use_tf32)
{
    try
    {
        switch(data_type)
        {
        case miopenHalf:
            return IsCKApplicable<DeviceOpGFwdPtrs<ck::half_t>, CKArgs>(*problem);
        case miopenFloat:
            if(use_tf32 &&
               IsCKApplicable<DeviceOpGFwdPtrs<float, ck::tf32_t>, CKArgs>(*problem))
                return true;
            return IsCKApplicable<DeviceOpGFwdPtrs<float>, CKArgs>(*problem);
        case miopenBFloat16:
            return IsCKApplicable<DeviceOpGFwdPtrs<ck::bhalf_t>, CKArgs>(*problem);
        case miopenInt8:
            return IsCKApplicable<DeviceOpGFwdPtrs<int8_t>, CKArgs>(*problem);
        default: return false;
        }
    }
    catch(...)
    {
        return false;
    }
}

extern "C" bool ckgrpconv_fwd_is_args_supported(
    const miopen::conv::ProblemDescription* problem,
    const char* kernel_id,
    miopenDataType_t data_type,
    bool use_tf32)
{
    try
    {
        const std::string kid(kernel_id ? kernel_id : "");
        switch(data_type)
        {
        case miopenHalf:
            return IsCKArgsSupported<DeviceOpGFwdPtrs<ck::half_t>, CKArgs>(*problem, kid);
        case miopenFloat:
            if(use_tf32 &&
               IsCKArgsSupported<DeviceOpGFwdPtrs<float, ck::tf32_t>, CKArgs>(*problem, kid))
                return true;
            return IsCKArgsSupported<DeviceOpGFwdPtrs<float>, CKArgs>(*problem, kid);
        case miopenBFloat16:
            return IsCKArgsSupported<DeviceOpGFwdPtrs<ck::bhalf_t>, CKArgs>(*problem, kid);
        case miopenInt8:
            return IsCKArgsSupported<DeviceOpGFwdPtrs<int8_t>, CKArgs>(*problem, kid);
        default: return false;
        }
    }
    catch(...)
    {
        return false;
    }
}

extern "C" size_t ckgrpconv_fwd_get_workspace_size(
    const miopen::conv::ProblemDescription* /*problem*/,
    miopenDataType_t /*data_type*/)
{
    // FWD grouped convolution CK kernels do not use split-k and never
    // require CK-level workspace.  Layout transform workspace (NCHW to NHWC)
    // is computed independently at the solver level via
    // GetWorkspaceSizeLayoutTransformConv().
    return 0;
}

extern "C" miopen::solver::ConvSolution* ckgrpconv_fwd_get_solution(
    const miopen::ExecutionContext* ctx,
    const miopen::conv::ProblemDescription* problem,
    const char* kernel_id,
    bool use_tf32)
{
    try
    {
        if(!ctx || !problem || !kernel_id)
            return nullptr;

        auto solution = MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryFwdNCHW<2,
                                                 false,
                                                 DeviceOpGFwdPtrs<T, TCompute>,
                                                 CKArgs,
                                                 miopen::conv::DataInvokeParams>(
                    *ctx, *problem, std::string(kernel_id));
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGFwdPtrs<T, TCompute>,
                                              CKArgs,
                                              miopen::conv::DataInvokeParams>(
                    *ctx, *problem, std::string(kernel_id));
            },
            use_tf32);
        return new miopen::solver::ConvSolution(std::move(solution));
    }
    catch(...)
    {
        return nullptr;
    }
}
