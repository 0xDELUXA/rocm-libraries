// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

/// Definition of the opaque handle used by the extern "C" interface.
/// Shared across all direction implementation files (fwd, bwd, wrw) and
/// defined in exactly one place to avoid ODR issues when linked into the
/// same shared library.
struct CKKernelListHandle
{
    std::vector<std::string> kernels;
};
