// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "bilateral/bilateral.hpp"

#include "backend/cuda/surface.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "backend/cuda/core1/kernels/defKernel.cu"

#include "backend/common/vectorOps.hpp"

#include "cuda/util.hpp"

#include <math.h>

static const float bilateralFilterRangeSigma = 0.005f;
static const float bilateralFilterSpatialSigma = 5.0f;
static const int bilateralFilterSpatialRadius = 16;  // int(3 * bilateralFilterSpatialSigma) + 1;

// TODODEPTH move these to vectorOps.hpp

inline __device__ __host__ float4 operator/(float4 v, float a) {
  return make_float4(v.x / a, v.y / a, v.z / a, v.w / a);
}

inline __device__ __host__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ __host__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ __host__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__inline__ __device__ __host__ float4 fmaxf(float4 a, float4 b) {
  return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __device__ __host__ float4 sqrt4(float4 val) {
  return make_float4(sqrtf(val.x), sqrtf(val.y), sqrtf(val.z), sqrtf(val.w));
}

inline __device__ __host__ float4 log4(float4 val) {
  return make_float4(logf(val.x), logf(val.y), logf(val.z), logf(val.w));
}

inline __device__ __host__ float4 exp4(float4 val) {
  return make_float4(expf(val.x), expf(val.y), expf(val.z), expf(val.w));
}

inline __device__ __host__ float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline __device__ __host__ int2 convert_int2(float2 val) { return make_int2((int)val.x, (int)val.y); }

inline __device__ float read_depth_vs(surface_t surf, float2 uv) {
  float val;
  surf2Dread(&val, surf, uv.x * sizeof(float), uv.y);
  return val;
}

inline __device__ float read_depth_vs(surface_t surf, int x, int y) {
  float val;
  surf2Dread(&val, surf, x * sizeof(float), y);
  return val;
}

#define sync_threads __syncthreads
#define CLK_GLOBAL_MEM_FENCE
#define __globalmem__

static const int CudaBlockSize = 16;

namespace VideoStitch {
namespace GPU {

#include "backend/common/bilateral/bilateral.gpu"

Status depthJointBilateralFilter(GPU::Surface& output, const GPU::Surface& input,
                                 const Core::SourceSurface& textureSurface, GPU::Stream& stream) {
  // template parameter in CUDA kernels assume that bilateralFilterSpatialRadius == int(3 * bilateralFilterSpatialSigma)
  // + 1
  assert(bilateralFilterSpatialRadius == int(3 * bilateralFilterSpatialSigma) + 1);

  PROPAGATE_FAILURE_CONDITION(updateGaussian(bilateralFilterSpatialSigma, bilateralFilterSpatialRadius),
                              Origin::Stitcher, ErrType::RuntimeError, "Failure in Gaussian kernel initialization");

  // Running a kernel that takes > 1s destabilizes the system
  // (Display manager resets or kernel panic)
  // As the current version is not optimised and works at full resolution it can take several seconds to complete
  // --> Tile the work. Each tile should complete in less than 1 second.
  const int numBlocks = 16;
  // Make sure texture width is a multiple of numBlocks
  const int paddedTexWidth = (int)Cuda::ceilDiv(input.width(), numBlocks) * numBlocks;
  const int paddedTexHeight = (int)Cuda::ceilDiv(input.height(), numBlocks) * numBlocks;
  for (int cx = 0; cx < numBlocks; cx++) {
    for (int cy = 0; cy < numBlocks; cy++) {
      const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
      const dim3 dimGrid((unsigned)Cuda::ceilDiv(paddedTexWidth / numBlocks, dimBlock.x),
                         (unsigned)Cuda::ceilDiv(paddedTexHeight / numBlocks, dimBlock.y), 1);
      depthBilateralFilterCrossShapedKernel<bilateralFilterSpatialRadius><<<dimGrid, dimBlock, 0, stream.get()>>>(
          output.get().surface(), (unsigned)output.width(), (unsigned)output.height(),
          textureSurface.pimpl->surface->get().texture(), input.get().surface(),
          1.0f / (2 * bilateralFilterRangeSigma * bilateralFilterRangeSigma), cx, cy, paddedTexWidth / numBlocks,
          paddedTexHeight / numBlocks);
      // Force synchronization after tile computation for system stability
      stream.synchronize();
    }
  }

  return CUDA_STATUS;
}

}  // namespace GPU
}  // namespace VideoStitch
