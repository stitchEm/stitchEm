// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <type_traits>

/** pseudo: `template <class S = T>, if T is not const`
 *  If T is const already, the specialization is discarded from the overload set instead of causing a compile error.
 *  SFINAE not fully supported by current CUDA and Visual Studio compilers.
 */
#if defined(_MSC_VER) || defined(__CUDACC__) || defined(__APPLE_CC__)
#define CLASS_S_EQ_T_ENABLE_IF_S_NON_CONST template <class S = T>
#else
#define CLASS_S_EQ_T_ENABLE_IF_S_NON_CONST \
  template <class S = T, typename std::enable_if<!std::is_const<S>{}>::type* = nullptr>
#endif  // _MSC_VER
