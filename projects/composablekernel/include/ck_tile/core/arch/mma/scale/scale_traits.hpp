// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
// #include "ck_tile/core/numeric/pk_fp6.hpp"

#include <cstdint>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#include <type_traits>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

template <typename T>
struct TypeToFlagValue;

template <>
struct TypeToFlagValue<fp8_t> // e4m3
{
    static constexpr std::uint8_t value = 0;
};

template <>
struct TypeToFlagValue<bf8_t> // e5m2
{
    static constexpr std::uint8_t value = 1;
};

// template <>
// struct TypeToFlagValue<pk_fp6_t<1>> // e2m3
// {
//     static constexpr std::uint8_t value = 2;
// };

// template <>
// struct TypeToFlagValue<bf6_t> // e3m2
// {
//     static constexpr std::uint8_t value = 3;
// };

template <>
struct TypeToFlagValue<pk_fp4_t> // e2m1
{
    static constexpr std::uint8_t value = 4;
};

template <typename T>
inline constexpr std::uint8_t TypeToFlagValue_v = TypeToFlagValue<T>::value;

template <typename ADataType, typename BDataType, std::uint8_t OPSELA = 0, std::uint8_t OPSELB = 0>
struct DefaultScaleMfmaCtrlFlags
{
    static constexpr std::uint8_t type_A  = TypeToFlagValue_v<ADataType>;
    static constexpr std::uint8_t type_B  = TypeToFlagValue_v<BDataType>;
    static constexpr std::uint8_t OPSEL_A = OPSELA;
    static constexpr std::uint8_t OPSEL_B = OPSELB;
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::type_A } -> std::convertible_to<int>;
    { CtrlFlags::type_B } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
