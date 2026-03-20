/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ReferenceValidator.hpp"
#include "ResultComparison.hpp"
#include "ResultReporter.hpp"
#include "TimingInstrumentation.hpp"

#include "Reference.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>
#include <cstring>

namespace TensileLite
{
    namespace Client
    {
        ReferenceValidator::ReferenceValidator(po::variables_map const&            args,
                                               std::shared_ptr<DataInitialization> dataInit)
            : m_dataInit(dataInit)
        {
            m_elementsToValidate = args["num-elements-to-validate"].as<int>();
            m_printValids        = args["print-valids"].as<bool>();
            m_printMax           = args["print-max"].as<int>();

            m_printTensorA             = args["print-tensor-a"].as<bool>();
            m_printTensorB             = args["print-tensor-b"].as<bool>();
            m_printTensorC             = args["print-tensor-c"].as<bool>();
            m_printTensorD             = args["print-tensor-d"].as<bool>();
            m_printTensorRef           = args["print-tensor-ref"].as<bool>();
            m_printTensorBias          = args["print-tensor-bias"].as<bool>();
            m_printTensorScaleAlphaVec = args["print-tensor-scale-alpha-vec"].as<bool>();
            m_printTensorAmaxD         = args["print-tensor-amaxd"].as<bool>();

            m_printAny = m_printTensorA || m_printTensorB || m_printTensorC || m_printTensorD
                         || m_printTensorRef || m_printTensorBias || m_printTensorAmaxD;

            m_enabled = m_elementsToValidate != 0 || m_printAny;

            {
                int numBenchmarks    = args["num-benchmarks"].as<int>();
                int numEnqPerSync    = args["num-enqueues-per-sync"].as<int>();
                int numSyncsPerBench = args["num-syncs-per-benchmark"].as<int>();
                m_noBenchmarkRuns
                    = (numBenchmarks == 0 || numEnqPerSync == 0 || numSyncsPerBench == 0);
            }

            if(m_enabled && m_noBenchmarkRuns)
                startWorker();
        }

        ReferenceValidator::~ReferenceValidator()
        {
            stopWorker();
        }

        bool ReferenceValidator::needMoreBenchmarkRuns() const
        {
            if(m_enabled && m_numBenchmarkRuns == 0)
                return true;

            return false;
        }

        void ReferenceValidator::preBenchmarkRun() {}

        void ReferenceValidator::postBenchmarkRun()
        {
            m_numBenchmarkRuns++;
        }

        void ReferenceValidator::preProblem(ContractionProblem* const problem)
        {
            if(m_enabled)
            {
                m_problem = problem;

                // Report problem context for timing correlation
                if(auto gemm = dynamic_cast<ContractionProblemGemm*>(problem))
                {
                    size_t M          = gemm->freeSizeA(0);
                    size_t N          = gemm->freeSizeB(0);
                    size_t K          = gemm->boundSize(0);
                    size_t batchCount = gemm->batchSize(0);
                    reportProblemContext(M, N, K, batchCount,
                                         TensileLite::ToString(gemm->a().dataType()),
                                         TensileLite::ToString(gemm->d().dataType()));
                }
                else if(auto grouped = dynamic_cast<ContractionProblemGroupedGemm*>(problem))
                {
                    size_t totalGemms = grouped->gemms.size();
                    for(size_t i = 0; i < totalGemms; i++)
                    {
                        auto&  g          = grouped->gemms[i];
                        size_t M          = g.freeSizeA(0);
                        size_t N          = g.freeSizeB(0);
                        size_t K          = g.boundSize(0);
                        size_t batchCount = g.batchSize(0);
                        reportGroupedProblemContext(i, totalGemms, M, N, K, batchCount,
                                                    TensileLite::ToString(g.a().dataType()),
                                                    TensileLite::ToString(g.d().dataType()));
                    }
                }

                if(m_slotCount > 0)
                {
                    // Pipeline active — wait for the next result in the ring.
                    std::unique_lock<std::mutex> lk(m_workerMtx);
                    auto                         waitStart = TimingClock::now();
                    auto&                        slot = m_slots[m_slotTail];
                    m_workerCv.wait(lk, [&] { return slot.done; });
                    m_referenceInputs = std::move(slot.inputs);
                    auto timings      = slot.timings;
                    slot.done         = false;
                    m_slotTail  = (m_slotTail + 1) % kQueueDepth;
                    m_slotCount--;
                    lk.unlock();
                    reportTiming("cpu_reference_gemm_wait",
                        std::chrono::duration<double, std::milli>(
                            TimingClock::now() - waitStart).count());
                    reportTiming("worker_pickup_delay", timings.pickupMs);
                    reportTiming("worker_task_duration", timings.solveCpuMs);
                }
                else
                {
                    {
                        ScopedTimer timer("cpu_data_init");
                        m_referenceInputs = m_dataInit->prepareCPUInputs(problem);
                    }
                    if(m_noBenchmarkRuns)
                    {
                        // Deep-copy + solve synchronously so m_cpuPtrs is
                        // free for the next problem's precomputation.
                        auto* gemm = dynamic_cast<ContractionProblemGemm const*>(problem);
                        if(gemm)
                        {
                            auto& src = dynamic_cast<ContractionInputs const&>(
                                *m_referenceInputs);
                            ScopedTimer timer("cpu_reference_gemm");
                            m_referenceInputs = deepCopyGemmInputs(*gemm, src);
                            SolveCPU(problem,
                                     m_referenceInputs.get(),
                                     m_elementsToValidate);
                        }
                    }
                    else
                    {
                        // Benchmark mode — launch async (simple overlap)
                        auto* refInputs          = m_referenceInputs.get();
                        int   elementsToValidate = m_elementsToValidate;
                        m_cpuGemmFuture          = std::async(std::launch::async,
                            [problem, refInputs, elementsToValidate]() {
                                SolveCPU(problem, refInputs, elementsToValidate);
                            });
                    }
                }
            }
        }

        void ReferenceValidator::preSolution(ContractionSolution* const solution)
        {
            m_validatedSolution = false;
            m_errorInSolution   = false;
            m_executedSolution  = false;
        }

        bool ReferenceValidator::needMoreRunsInSolution() const
        {
            if(m_enabled && !m_validatedSolution)
                return true;

            return false;
        }

        size_t ReferenceValidator::numWarmupRuns()
        {
            if(m_enabled && !m_validatedSolution)
                return 1;

            return 0;
        }

        void ReferenceValidator::setNumWarmupRuns(size_t count) {}

        void ReferenceValidator::preWarmup() {}

        void ReferenceValidator::postWarmup(TimingEvents const& startEvents,
                                            TimingEvents const& stopEvents,
                                            hipStream_t const&  stream)
        {
            m_executedSolution = true;
        }

        bool ReferenceValidator::validateSolution(std::shared_ptr<ProblemInputs> inputs)
        {
            if(!m_enabled)
                return false;

            bool rv = false;

            if(m_elementsToValidate != 0)
            {
                if(auto problems = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
                {
                    auto reference
                        = dynamic_cast<ContractionGroupedInputs const&>(*m_referenceInputs);
                    auto result = dynamic_cast<ContractionGroupedInputs const&>(*inputs);
                    rv          = true;
                    for(size_t j = 0; j < problems->gemms.size(); j++)
                    {
                        rv &= validate(problems->gemms[j], reference.grouped[j], result.grouped[j]);
                    }
                }
                else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                {
                    auto reference = dynamic_cast<ContractionInputs const&>(*m_referenceInputs);
                    auto result    = dynamic_cast<ContractionInputs const&>(*inputs);
                    rv             = validate(*problem, reference, result);
                }
                else
                {
                    throw std::runtime_error("Failed to cast to any ContractionProblem.");
                }
            }

            return rv;
        }

        void ReferenceValidator::validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                                 TimingEvents const&            startEvents,
                                                 TimingEvents const&            stopEvents)
        {
            if(m_enabled && !m_validatedSolution)
            {
                if(m_cpuGemmFuture.valid())
                {
                    auto waitStart = TimingClock::now();
                    m_cpuGemmFuture.get();
                    reportTiming("cpu_reference_gemm_wait",
                        std::chrono::duration<double, std::milli>(
                            TimingClock::now() - waitStart).count());
                }

                ScopedTimer timer("validate_reference");
                validateSolution(inputs);
                m_validatedSolution = true;
            }
        }

        bool ReferenceValidator::checkResults(TensorDescriptor const& tensor,
                                              void const*             refPtr,
                                              void const*             resPtr,
                                              size_t                  maxElements,
                                              bool                    isgpu,
                                              size_t                  validationStride,
                                              double                  threshold)
        {
            bool rv = false;
            switch(tensor.dataType())
            {
            case rocisa::DataType::Float:
            {
                rv = checkResultsTyped(tensor,
                                       (float const*)refPtr,
                                       (float const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Double:
            {
                rv = checkResultsTyped(tensor,
                                       (double const*)refPtr,
                                       (double const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::ComplexFloat:
            {
                rv = checkResultsTyped(tensor,
                                       (std::complex<float> const*)refPtr,
                                       (std::complex<float> const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::ComplexDouble:
            {
                rv = checkResultsTyped(tensor,
                                       (std::complex<double> const*)refPtr,
                                       (std::complex<double> const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Half:
            {
                rv = checkResultsTyped(tensor,
                                       (Half const*)refPtr,
                                       (Half const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Float8:
            {
                rv = checkResultsTyped(tensor,
                                       (Float8 const*)refPtr,
                                       (Float8 const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::BFloat8:
            {
                rv = checkResultsTyped(tensor,
                                       (BFloat8 const*)refPtr,
                                       (BFloat8 const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Float8_fnuz:
            {
                rv = checkResultsTyped(tensor,
                                       (Float8_fnuz const*)refPtr,
                                       (Float8_fnuz const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::BFloat8_fnuz:
            {
                rv = checkResultsTyped(tensor,
                                       (BFloat8_fnuz const*)refPtr,
                                       (BFloat8_fnuz const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Int8x4:
            {
                throw std::runtime_error("Unsupported validator data type Int8x4 for output.");
            }
            break;
            case rocisa::DataType::Int32:
            {
                rv = checkResultsTyped(tensor,
                                       (int32_t const*)refPtr,
                                       (int32_t const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::BFloat16:
            {
                rv = checkResultsTyped(tensor,
                                       (BFloat16 const*)refPtr,
                                       (BFloat16 const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Int8:
            {
                rv = checkResultsTyped(tensor,
                                       (int8_t const*)refPtr,
                                       (int8_t const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            default:
                throw std::runtime_error("Unsupported validator data type");
            }
            if(rv)
            {
                std::cout << "Check failed in output tensor: " << tensor << std::endl;
            }
            return rv;
        }

        bool ReferenceValidator::validate(ContractionProblemGemm const& problem,
                                          ContractionInputs const&      reference,
                                          ContractionInputs const&      result)
        {
            if(problem.tensors().empty())
                return false;

            bool rv = true;

            if(m_printAny)
                printTensors(problem, reference, result);

            auto k = problem.transA() ? problem.a().sizes().at(0) : problem.a().sizes().at(1);
            bool isTF32 = (problem.f32XdlMathOp() == rocisa::DataType::XFloat32);
            bool isTF32x1 = (problem.computeInputType() == rocisa::DataType::BFloat16
                && problem.computeType() == rocisa::DataType::Float);
            double threshold = -1.0;
            if (isTF32) {
                threshold = 0.01 * sqrt(double(k));
            } else if (isTF32x1) {
                threshold = 0.3 * sqrt(double(k));
            }

            for(size_t i = 0; i < problem.tensors().size(); i++)
            {
                auto& tensor = problem.tensors()[i];
                if(!tensor.isOutput())
                    continue;

                size_t validationStride = 1;
                if(m_elementsToValidate > 0 && m_elementsToValidate < tensor.totalLogicalElements())
                    validationStride
                        = NextPrime(tensor.totalAllocatedElements() / m_elementsToValidate);

                void const* refPtr = nullptr;
                void const* resPtr = nullptr;
                switch(static_cast<ContractionProblemGemm::TENSOR>(i))
                {
                case ContractionProblemGemm::TENSOR::A:
                {
                    refPtr = reference.a;
                    resPtr = result.a;
                }
                break;
                case ContractionProblemGemm::TENSOR::B:
                {
                    refPtr = reference.b;
                    resPtr = result.b;
                }
                break;
                case ContractionProblemGemm::TENSOR::C:
                {
                    refPtr = reference.c;
                    resPtr = result.c;
                }
                break;
                case ContractionProblemGemm::TENSOR::D:
                {
                    refPtr = reference.d;
                    resPtr = result.d;
                }
                break;
                case ContractionProblemGemm::TENSOR::E:
                {
                    refPtr = reference.e;
                    resPtr = result.e;
                }
                break;
                case ContractionProblemGemm::TENSOR::BIAS:
                {
                    refPtr = reference.bias;
                    resPtr = result.bias;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEA:
                {
                    refPtr = reference.scaleA;
                    resPtr = result.scaleA;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEB:
                {
                    refPtr = reference.scaleB;
                    resPtr = result.scaleB;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEC:
                {
                    refPtr = reference.scaleC;
                    resPtr = result.scaleC;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALED:
                {
                    refPtr = reference.scaleD;
                    resPtr = result.scaleD;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEALPHAVEC:
                {
                    refPtr = reference.scaleAlphaVec;
                    resPtr = result.scaleAlphaVec;
                }
                case ContractionProblemGemm::TENSOR::Synchronizer:
                {
                    refPtr = reference.Synchronizer;
                    resPtr = result.Synchronizer;
                }
                case ContractionProblemGemm::TENSOR::AMAXD:
                {
                    refPtr = reference.amaxD;
                    resPtr = result.amaxD;
                }

                break;
                default:
                    throw std::runtime_error("Unrecognized output tensor.");
                }

                if(Debug::Instance().printTensorInfo())
                    std::cout << "Validating tensor " << tensor.getName() << ", cpu pointer "
                              << refPtr << ", gpu pointer " << resPtr
                              << ", size = " << result.maxElements[i] << std::endl;
                
                rv &= checkResults(
                    tensor, refPtr, resPtr, result.maxElements[i], result.gpu, validationStride, threshold);
            }
            return rv;
        }

        void ReferenceValidator::allocateResultBuffer(size_t bytes)
        {
            if(m_cpuResultBufferSize == bytes)
                return;
            m_cpuResultBuffer.reset();

            uint8_t* buffer;
            HIP_CHECK_EXC(hipHostMalloc(&buffer, bytes, 0));
            m_cpuResultBuffer.reset(buffer, hipFree);
            m_cpuResultBufferSize = bytes;
        }

        void ReferenceValidator::printTensors(ContractionProblemGemm const& problem,
                                              ContractionInputs const&      reference,
                                              ContractionInputs const&      result)
        {
            size_t requiredBufferSize = 0;

            std::cout << "reference alpha: " << ToString(reference.alpha)
                      << ", beta: " << ToString(reference.beta) << std::endl;
            std::cout << "result    alpha: " << ToString(result.alpha)
                      << ", beta: " << ToString(result.beta) << std::endl;

            if(m_printTensorA)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.a().totalAllocatedBytes());
            if(m_printTensorB)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.b().totalAllocatedBytes());
            if(m_printTensorC)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.c().totalAllocatedBytes());
            if(m_printTensorD)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.d().totalAllocatedBytes());
            if(m_printTensorRef)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.d().totalAllocatedBytes());
            if(m_printTensorBias)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.bias().totalAllocatedBytes());
            if(m_printTensorScaleAlphaVec)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.scaleAlphaVec().totalAllocatedBytes());
            if(m_printTensorAmaxD)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.amaxd().totalAllocatedBytes());

            if(m_cpuResultBufferSize < requiredBufferSize)
                allocateResultBuffer(requiredBufferSize);

            if(m_printTensorA)
            {
                auto a = problem.a();
                m_reporter->logTensor(
                    LogLevel::Verbose, "A", reference.a, problem.a(), reference.a);
                if(problem.sparse() && problem.sparse() != 2)
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "Compressed A",
                                          reference.compressed,
                                          problem.compressed(),
                                          reference.compressed);
                }
            }

            if(m_printTensorB)
            {
                auto b = problem.b();
                m_reporter->logTensor(
                    LogLevel::Verbose, "B", reference.b, problem.b(), reference.b);
                if(problem.sparse() && problem.sparse() == 2)
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "Compressed B",
                                          reference.compressed,
                                          problem.compressed(),
                                          reference.compressed);
                }
            }

            if(m_printTensorA || m_printTensorB)
            {
                if(problem.sparse())
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "Metadata",
                                          reference.metadata,
                                          problem.metadata(),
                                          reference.metadata);
                }
            }

            if(result.c == result.d && (m_printTensorC || m_printTensorD))
            {
                // If the pointers are the same, only print the buffer once.
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.c,
                                        problem.c().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(
                    LogLevel::Verbose, "C_D", m_cpuResultBuffer.get(), problem.c(), result.c);
            }
            else
            {
                if(m_printTensorC)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                            result.c,
                                            problem.c().totalAllocatedBytes(),
                                            hipMemcpyDeviceToHost));
                    m_reporter->logTensor(
                        LogLevel::Verbose, "C", m_cpuResultBuffer.get(), problem.c(), result.c);
                }

                if(m_printTensorD)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                            result.d,
                                            problem.d().totalAllocatedBytes(),
                                            hipMemcpyDeviceToHost));
                    m_reporter->logTensor(
                        LogLevel::Verbose, "D", m_cpuResultBuffer.get(), problem.d(), result.d);
                }
            }

            if(m_printTensorRef)
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "Ref", reference.d, problem.d(), reference.d);
            }

            if(m_printTensorBias)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.bias,
                                        problem.bias().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "bias",
                                      m_cpuResultBuffer.get(),
                                      problem.bias(),
                                      result.bias);
            }
            if(m_printTensorScaleAlphaVec)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.scaleAlphaVec,
                                        problem.scaleAlphaVec().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "scaleAlphaVec",
                                      m_cpuResultBuffer.get(),
                                      problem.scaleAlphaVec(),
                                      result.scaleAlphaVec);
            }

            if(m_printTensorAmaxD)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.amaxD,
                                        problem.amaxd().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "AmaxD Ref",
                                      reference.amaxD,
                                      problem.amaxd(),
                                      reference.amaxD);
                m_reporter->logTensor(LogLevel::Verbose,
                                      "AmaxD GPU",
                                      m_cpuResultBuffer.get(),
                                      problem.amaxd(),
                                      result.amaxD);
            }
        }

        template <typename ValidType, typename Comparator>
        void forEachElement(TensorDescriptor const& tensor,
                            ValidType const*        reference,
                            ValidType const*        resultData,
                            size_t                  validationStride,
                            Comparator&             compare)
        {
            if(validationStride == 1)
            {
                std::vector<size_t> coord(tensor.dimensions());
                size_t outerCount
                    = CoordCount(tensor.sizes().begin() + 1, tensor.sizes().end());

                size_t       elemNumberBase = 0;
                const size_t innerDimSize   = tensor.sizes()[0];
                const size_t initialStride  = tensor.strides()[0];

                for(size_t i = 0; i < outerCount; i++)
                {
                    CoordNumbered(i,
                                  coord.begin() + 1,
                                  coord.end(),
                                  tensor.sizes().begin() + 1,
                                  tensor.sizes().end());
                    size_t baseElemIndex = tensor.index(coord);

                    for(size_t j = 0; j < innerDimSize; j++)
                    {
                        size_t elemIndex  = baseElemIndex + (j * initialStride);
                        size_t elemNumber = elemNumberBase + j;

                        compare(reference[elemIndex], resultData[elemIndex],
                                elemIndex, elemNumber);
                    }
                    elemNumberBase += innerDimSize;
                }
            }
            else
            {
                std::vector<size_t> coord(tensor.dimensions());
                for(size_t elemNumber = 0;
                    elemNumber < tensor.totalLogicalElements();
                    elemNumber += validationStride)
                {
                    CoordNumbered(elemNumber,
                                  coord.begin(),
                                  coord.end(),
                                  tensor.sizes().begin(),
                                  tensor.sizes().end());
                    size_t elemIndex = tensor.index(coord);

                    compare(reference[elemIndex], resultData[elemIndex],
                            elemIndex, elemNumber);
                }
            }
        }

        template <typename ValidType>
        bool ReferenceValidator::checkResultsTyped(TensorDescriptor const& tensor,
                                                   ValidType const*        reference,
                                                   ValidType const*        result,
                                                   size_t                  maxElement,
                                                   bool                    isgpu,
                                                   size_t                  validationStride,
                                                   double                  threshold)
        {
            size_t elementsToCopy       = tensor.totalAllocatedElements();
            size_t elementsOffsetToCopy = 0;
            size_t elementsBeforeData   = 0;
            size_t elementsAfterData    = 0;

            BoundsCheckMode boundsCheck = m_dataInit->getCurBoundsCheck();
            if(boundsCheck == BoundsCheckMode::NaN)
                elementsToCopy = maxElement;
            size_t bytesToCopy = elementsToCopy * sizeof(ValidType);

            if(m_cpuResultBufferSize < bytesToCopy)
                allocateResultBuffer(bytesToCopy);

            auto copykind = isgpu ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;

            {
                ScopedTimer timer("validate_gpu_readback");
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(), result, bytesToCopy, copykind));
            }

            if(boundsCheck == BoundsCheckMode::NaN)
            {
                ptrdiff_t bPadding = maxElement - tensor.totalAllocatedElements();
                elementsBeforeData = bPadding / 2;
                elementsAfterData
                    = elementsToCopy - (tensor.totalAllocatedElements() + elementsBeforeData);
            }
            // If there was extra data allocated before the tensor to do bounds
            // checking, resultBuffer is the whole allocation, while resultData
            // points directly to the result.
            ValidType const* resultBuffer
                = reinterpret_cast<ValidType const*>(m_cpuResultBuffer.get());
            ValidType const* resultData      = resultBuffer + elementsBeforeData;
            ValidType const* resultAfterData = resultData + tensor.totalAllocatedElements();

            FastPointwiseComparison<ValidType> compareValid(m_printMax > 0, threshold);
            InvalidComparison<ValidType>   compareInvalid(m_printMax, m_printMax > 0);

            size_t boundsCheckElements = 0;

            {
                ScopedTimer timer("validate_element_comparison");

                for(ptrdiff_t i = 0; i < elementsBeforeData; i++)
                {
                    boundsCheckElements++;
                    compareInvalid.before(resultBuffer[i], i, elementsBeforeData);
                }

                forEachElement(tensor, reference, resultData, validationStride, compareValid);

                if(boundsCheck == BoundsCheckMode::NaN && validationStride == 1)
                {
                    std::vector<size_t> coord(tensor.dimensions());
                    size_t outerCount
                        = CoordCount(tensor.sizes().begin() + 1, tensor.sizes().end());
                    size_t       prevBaseIndex = 0;
                    const size_t innerDimSize  = tensor.sizes()[0];

                    for(size_t i = 0; i < outerCount; i++)
                    {
                        CoordNumbered(i,
                                      coord.begin() + 1,
                                      coord.end(),
                                      tensor.sizes().begin() + 1,
                                      tensor.sizes().end());
                        size_t baseElemIndex = tensor.index(coord);

                        if(baseElemIndex != 0
                           && baseElemIndex != prevBaseIndex + innerDimSize)
                        {
                            for(auto innerIndex = prevBaseIndex + innerDimSize;
                                innerIndex < baseElemIndex;
                                innerIndex++)
                            {
                                compareInvalid.inside(
                                    resultData[innerIndex], innerIndex, baseElemIndex);
                            }
                        }
                        prevBaseIndex = baseElemIndex;
                    }
                }

                for(ptrdiff_t i = 0; i < elementsAfterData; i++)
                {
                    compareInvalid.after(resultAfterData[i], i, elementsAfterData);
                }
            }

            if(boundsCheckElements > 0)
                std::cout << "Performed bounds check on " << boundsCheckElements << " elements ("
                          << elementsBeforeData << " before data)" << std::endl;

            if((compareValid.errorCount() > 0 || m_printValids) && m_printMax > 0)
            {
                ScopedTimer timer("validate_mismatch_printing");

                PointwiseComparison<ValidType> comparePrint(
                    m_printValids, m_printMax, false, threshold);

                forEachElement(tensor, reference, resultData, validationStride, comparePrint);
            }

            compareValid.report();
            compareInvalid.report();

            if(compareValid.error() || compareInvalid.error())
            {
                m_errorInSolution = true;
                m_error           = true;

                return true;
            }

            return false;
        }

        void ReferenceValidator::postSolution()
        {
            ScopedTimer timer("post_solution_validation");
            if(!m_executedSolution)
                return;

            if(m_enabled && !m_validatedSolution)
                return;

            if(m_elementsToValidate != 0)
            {
                if(m_errorInSolution)
                {
                    m_errorsReported++;
                    m_reporter->report(ResultKey::Validation, "FAILED");
                }
                else
                    m_reporter->report(ResultKey::Validation, "PASSED");
            }
            else
            {
                m_reporter->report(ResultKey::Validation, "NO_CHECK");
            }

            m_errorInSolution = false;
        }

        void ReferenceValidator::postProblem() {}

        // ---------------------------------------------------------------
        // Deep-copy + pipeline helpers
        // ---------------------------------------------------------------

        std::shared_ptr<ProblemInputs> ReferenceValidator::deepCopyGemmInputs(
            ContractionProblemGemm const& problem,
            ContractionInputs const&      src)
        {
            using T = ContractionProblemGemm::TENSOR;

            auto const& tensors   = problem.tensors();
            int const   numTensors = static_cast<int>(tensors.size());

            // 1. Compute total byte size and per-tensor offsets.
            size_t              totalBytes = 0;
            std::vector<size_t> offsets(T::TENSOR_COUNT, 0);
            for(int i = 0; i < numTensors && i < T::TENSOR_COUNT; i++)
            {
                offsets[i] = totalBytes;
                totalBytes += tensors[i].totalAllocatedBytes();
            }

            // 2. Allocate a single contiguous buffer.
            auto     backing = std::make_shared<std::vector<uint8_t>>(totalBytes);
            uint8_t* base    = backing->data();

            // 3. Helper: get the source pointer for a tensor index.
            auto srcField = [&](int idx) -> void const* {
                switch(static_cast<T>(idx))
                {
                case T::A:             return src.a;
                case T::B:             return src.b;
                case T::C:             return src.c;
                case T::D:             return src.d;
                case T::E:             return src.e;
                case T::BIAS:          return src.bias;
                case T::SCALEA:        return src.scaleA;
                case T::SCALEB:        return src.scaleB;
                case T::SCALEC:        return src.scaleC;
                case T::SCALED:        return src.scaleD;
                case T::SCALEALPHAVEC: return src.scaleAlphaVec;
                case T::METADATA:      return src.metadata;
                case T::Synchronizer:  return src.Synchronizer;
                case T::AMAXD:         return src.amaxD;
                case T::COMPRESSED:    return src.compressed;
                default:               return nullptr;
                }
            };

            // 4. Build the destination ContractionInputs.
            auto* dst            = new ContractionInputs();
            dst->alpha           = src.alpha;
            dst->beta            = src.beta;
            dst->activationArgs  = src.activationArgs;
            dst->maxElements     = src.maxElements;
            dst->workspaceSize   = src.workspaceSize;
            dst->gpu             = false;

            // 5. Copy each tensor and set the dst pointer.
            auto setDstField = [&](int idx, void* ptr) {
                switch(static_cast<T>(idx))
                {
                case T::A:             dst->a             = ptr; break;
                case T::B:             dst->b             = ptr; break;
                case T::C:             dst->c             = ptr; break;
                case T::D:             dst->d             = ptr; break;
                case T::E:             dst->e             = ptr; break;
                case T::BIAS:          dst->bias          = ptr; break;
                case T::SCALEA:        dst->scaleA        = ptr; break;
                case T::SCALEB:        dst->scaleB        = ptr; break;
                case T::SCALEC:        dst->scaleC        = ptr; break;
                case T::SCALED:        dst->scaleD        = ptr; break;
                case T::SCALEALPHAVEC: dst->scaleAlphaVec = ptr; break;
                case T::METADATA:      dst->metadata      = (unsigned char*)ptr; break;
                case T::Synchronizer:  dst->Synchronizer  = ptr; break;
                case T::AMAXD:         dst->amaxD         = ptr; break;
                case T::COMPRESSED:    dst->compressed    = ptr; break;
                default: break;
                }
            };

            for(int i = 0; i < numTensors && i < T::TENSOR_COUNT; i++)
            {
                void const* sp = srcField(i);
                if(!sp)
                    continue;
                size_t bytes = tensors[i].totalAllocatedBytes();
                if(bytes == 0)
                    continue;
                std::memcpy(base + offsets[i], sp, bytes);
                setDstField(i, base + offsets[i]);
            }

            // 6. Return shared_ptr that captures the backing buffer.
            return std::shared_ptr<ProblemInputs>(
                dst, [backing](ProblemInputs* p) { delete p; });
        }

        void ReferenceValidator::startPrecomputeForNextProblem(
            ContractionProblem** problems, int count)
        {
            if(!m_enabled || !m_noBenchmarkRuns)
                return;

            for(int i = 0; i < count; i++)
            {
                auto* gemmProblem
                    = dynamic_cast<ContractionProblemGemm const*>(problems[i]);
                if(!gemmProblem)
                    continue;

                std::shared_ptr<ProblemInputs> snapshot;
                {
                    ScopedTimer timer("cpu_data_init");
                    snapshot = m_dataInit->prepareCPUInputs(problems[i]);
                }

                // Deep-copy on main thread (~1ms) so buffers are free
                // for the next iteration.
                auto& src     = dynamic_cast<ContractionInputs const&>(*snapshot);
                auto deepCopy = deepCopyGemmInputs(*gemmProblem, src);

                // Post into the next ring slot.
                {
                    std::lock_guard<std::mutex> lk(m_workerMtx);
                    auto& slot              = m_slots[m_slotHead];
                    slot.inputs             = std::move(deepCopy);
                    slot.problem            = problems[i];
                    slot.elementsToValidate = m_elementsToValidate;
                    slot.postTime           = TimingClock::now();
                    slot.ready              = true;
                    slot.done               = false;
                    m_slotHead = (m_slotHead + 1) % kQueueDepth;
                    m_slotCount++;
                }
                m_workerCv.notify_one();
            }

        }

        int ReferenceValidator::pendingCount() const
        {
            return m_slotCount;
        }

        // ---- Persistent worker thread ----------------------------------------

        void ReferenceValidator::startWorker()
        {
            m_workerStop = false;
            m_worker     = std::thread(&ReferenceValidator::workerLoop, this);
        }

        void ReferenceValidator::stopWorker()
        {
            if(!m_worker.joinable())
                return;
            {
                std::lock_guard<std::mutex> lk(m_workerMtx);
                m_workerStop = true;
            }
            m_workerCv.notify_one();
            m_worker.join();
        }

        void ReferenceValidator::workerLoop()
        {
            using ms = std::chrono::duration<double, std::milli>;

            // Use fewer OMP threads than the synchronous path
            // to leave cores free for the main thread.
            g_solveCpuOmpThreads = std::max(1, kSolveCpuMaxOmpThreads - 2);

            while(true)
            {
                int slotIdx;
                {
                    std::unique_lock<std::mutex> lk(m_workerMtx);
                    m_workerCv.wait(lk, [&] {
                        return m_workerStop || m_slots[m_workerIdx].ready;
                    });
                    if(m_workerStop)
                        return;
                    slotIdx = m_workerIdx;
                    m_slots[slotIdx].ready = false;
                    m_workerIdx = (m_workerIdx + 1) % kQueueDepth;
                }

                auto& slot      = m_slots[slotIdx];
                auto  pickupTime = TimingClock::now();

                SolveCPU(slot.problem, slot.inputs.get(), slot.elementsToValidate);

                auto doneTime = TimingClock::now();
                {
                    std::lock_guard<std::mutex> lk(m_workerMtx);
                    slot.timings.pickupMs   = ms(pickupTime - slot.postTime).count();
                    slot.timings.solveCpuMs = ms(doneTime - pickupTime).count();
                    slot.done = true;
                }
                m_workerCv.notify_one();
                // Loop immediately — if another slot is ready, pick it up.
            }
        }

        void ReferenceValidator::finalizeReport() {}

        int ReferenceValidator::error() const
        {
            return m_errorsReported;
        }
    } // namespace Client
} // namespace TensileLite
