// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

/**
 * Returns int(ceil(v/d))
 */
template <typename IntT>
inline int64_t ceilDiv(int64_t v, IntT d) {
  const int64_t res = v / (int64_t)d;
  return res + (int64_t)(v - res * (int64_t)d > 0);  // add one if the remainder is nonzero
}

template <typename IntT>
inline IntT ceil(IntT v, IntT d) {
  return ceilDiv(v, d) * d;
}
