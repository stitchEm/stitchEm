// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/core1/strip.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"
#include "../core/transformStack.cu"

#include "cuda/util.hpp"
#include "core/geoTransform.hpp"
#include "core/transformGeoParams.hpp"

#include <limits>

namespace VideoStitch {
namespace Core {
enum Direction { Vertical, Horizontal };

template <Convert2D3DFnT toSphere, Direction direction>
__global__ void stripKernel(unsigned char* dstBuf, int2 dstDim, float2 distCenter, const vsDistortion distortion,
                            const float2 inputScale, float min, float max) {
  int2 dst = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (dst.x < dstDim.x && dst.y < dstDim.y) {
    /* extract only if in the strip */
    float2 sph;
    sph.x = (float)dst.x - dstDim.x / 2;
    sph.y = (float)dst.y - dstDim.y / 2;
    sph.x -= distCenter.x;
    sph.y -= distCenter.y;

    // inverseRadial(sph, radial0, radial1, radial2, radial3, radial4);
    sph.x /= inputScale.x;
    sph.y /= inputScale.y;

    /* take input projection into account*/
    float3 pt = toSphere(sph);

    if (direction == Horizontal) {
      vsfloat3x3 rot;
      rot.values[0][0] = 0.0f;
      rot.values[0][1] = -1.0f;
      rot.values[0][2] = 0.0f;
      rot.values[1][0] = 1.0f;
      rot.values[1][1] = 0.0f;
      rot.values[1][2] = 0.0f;
      rot.values[2][0] = 0.0f;
      rot.values[2][1] = 0.0f;
      rot.values[2][2] = 1.0f;
      pt = rotateSphere(pt, rot);
    }

    sph = SphereToErect(pt);
    if (sph.x < min || sph.x > max) {
      dstBuf[dstDim.x * dst.y + dst.x] = 1;
    }
  }
}

#define STRIP_KERNEL(transformFn, direction)                             \
  stripKernel<transformFn, direction><<<dimGrid, dimBlock, 0, stream>>>( \
      dst.get(), dstDim, distCenter, geoParams.getDistortion(), inputScale, min, max);

Status hStrip(GPU::Buffer<unsigned char> dst, std::size_t dstWidth, std::size_t dstHeight, float min, float max,
              InputDefinition::Format fmt, float distCenterX, float distCenterY, const TransformGeoParams& geoParams,
              const float2& inputScale, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
  int2 dstDim = make_int2((int)dstWidth, (int)dstHeight);

  // hardcode some stuff while we don't know how to evaluate the rig
  float2 distCenter = make_float2(distCenterX, distCenterY);
  switch (fmt) {
    case InputDefinition::Format::Rectilinear:
      STRIP_KERNEL(RectToSphere, Horizontal);
      break;
    case InputDefinition::Format::Equirectangular:
      STRIP_KERNEL(ErectToSphere, Horizontal);
      break;
    case InputDefinition::Format::CircularFisheye:
    case InputDefinition::Format::FullFrameFisheye:
    case InputDefinition::Format::CircularFisheye_Opt:
    case InputDefinition::Format::FullFrameFisheye_Opt:
      STRIP_KERNEL(FisheyeToSphere, Horizontal);
      break;
  }
  return CUDA_STATUS;
}

Status vStrip(GPU::Buffer<unsigned char> dst, std::size_t dstWidth, std::size_t dstHeight, float min, float max,
              InputDefinition::Format fmt, float distCenterX, float distCenterY, const TransformGeoParams& geoParams,
              const float2& inputScale, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
  int2 dstDim = make_int2((int)dstWidth, (int)dstHeight);

  // hardcode some stuff while we don't know how to evaluate the rig
  float2 distCenter = make_float2(distCenterX, distCenterY);
  switch (fmt) {
    case InputDefinition::Format::Rectilinear:
      STRIP_KERNEL(RectToSphere, Vertical);
      break;
    case InputDefinition::Format::Equirectangular:
      STRIP_KERNEL(ErectToSphere, Vertical);
      break;
    case InputDefinition::Format::CircularFisheye:
    case InputDefinition::Format::FullFrameFisheye:
    case InputDefinition::Format::CircularFisheye_Opt:
    case InputDefinition::Format::FullFrameFisheye_Opt:
      STRIP_KERNEL(FisheyeToSphere, Vertical);
      break;
  }
  return CUDA_STATUS;
}
}  // namespace Core
}  // namespace VideoStitch
