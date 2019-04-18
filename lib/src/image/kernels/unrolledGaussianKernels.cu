// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef UNROLLED_GAUSSIAN_KERNEL_HPP_
#define UNROLLED_GAUSSIAN_KERNEL_HPP_

#include "backend/common/imageOps.hpp"
#include "backend/common/vectorOps.hpp"

namespace VideoStitch {
namespace Image {

inline __device__ void addToAccumulatorsWeightedRGBA(const int32_t *&argb, int32_t &tr, int32_t &tg, int32_t &tb,
                                                     int32_t &acc, int32_t weight) {
  int32_t isSolid = (*argb++) * weight;
  tr += isSolid * (*argb++);
  tg += isSolid * (*argb++);
  tb += isSolid * (*argb++);
  acc += isSolid;
}

__device__ inline uint32_t unrolledGaussianKernel1(const int32_t *col) {
  int32_t tr = 0;
  int32_t tg = 0;
  int32_t tb = 0;
  int32_t acc = 0;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  int32_t isSolid = *col;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 2);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);
}

__device__ inline uint32_t unrolledGaussianKernel2(const int32_t *col) {
  int32_t tr = 0;
  int32_t tg = 0;
  int32_t tb = 0;
  int32_t acc = 0;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 4);
  int32_t isSolid = *col;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 6);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 4);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);
}

__device__ inline uint32_t unrolledGaussianKernel3(const int32_t *col) {
  int32_t tr = 0;
  int32_t tg = 0;
  int32_t tb = 0;
  int32_t acc = 0;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 6);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 15);
  int32_t isSolid = *col;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 20);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 15);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 6);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);
}

__device__ inline uint32_t unrolledGaussianKernel4(const int32_t *col) {
  int32_t tr = 0;
  int32_t tg = 0;
  int32_t tb = 0;
  int32_t acc = 0;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 8);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 28);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 56);
  int32_t isSolid = *col;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 70);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 56);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 28);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 8);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);
}

__device__ inline uint32_t unrolledGaussianKernel5(const int32_t *col) {
  int32_t tr = 0;
  int32_t tg = 0;
  int32_t tb = 0;
  int32_t acc = 0;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 10);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 45);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 120);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 210);
  int32_t isSolid = *col;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 252);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 210);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 120);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 45);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 10);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);
}

__device__ inline uint32_t unrolledGaussianKernel6(const int32_t *col) {
  int32_t tr = 0;
  int32_t tg = 0;
  int32_t tb = 0;
  int32_t acc = 0;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 12);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 66);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 220);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 495);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 792);
  int32_t isSolid = *col;
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 924);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 792);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 495);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 220);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 66);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 12);
  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, 1);
  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);
}
}  // namespace Image
}  // namespace VideoStitch
#endif
