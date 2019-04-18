// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

#include <cuda_runtime.h>

#include <cassert>

namespace VideoStitch {
namespace Cuda {

struct Stream {
  Stream() : s(0) {}
  explicit Stream(cudaStream_t s) : s(s) {}

  cudaStream_t s;
};

/**
 * Returns int(ceil(v/d))
 */
template <typename IntT>
inline int64_t ceilDiv(int64_t v, IntT d) {
  const int64_t res = v / (int64_t)d;
  return res + (int64_t)(v - res * (int64_t)d > 0);  // add one is the remainder is nonzero
}

/**
 * Given a flat buffer; compute a 2D grid of threads lauchable by cuda.
 * This makes it possible to lauch grids with more than 65k blocks, which cannot be 1D with version < 3.
 * @param size Size of the flat buffer.
 * @param blockSize Number of threads in a block.
 *
 * Worst case uselessly launched blocks is:
 *    n(k) = 2 n(k-1) + 2^k  =>  n(k) = (k + 1) 2^k
 * with k the number of divisions by 2.
 * At the same time, the number of launched blocks is around:
 *    MAXGRIDDIM * 2^k
 * So the proportion of useless blocks is:
 *    (k + 1) 2^k / (MAXGRIDDIM * 2^k) = (k + 1) / MAXGRIDDIM
 * Which remains reasonable. If we go to incredibly large images, one idea would be to
 * factor as primes and build from the bottom up to avoid running empty blocks.
 *   l = prime_factors(ceilDiv(size))
 *   dim1 = 1
 *   while (dim1 * l.back() < MAXGRIDDIM) {
 *     dim1 *= l.back();
 *     l.pop();
 *   }
 */
dim3 compute2DGridForFlatBuffer(int64_t size, unsigned blockSize);

}  // namespace Cuda
}  // namespace VideoStitch
