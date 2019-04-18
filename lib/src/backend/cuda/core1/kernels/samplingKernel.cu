// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/imageOps.hpp"
#include "image/kernels/sharedUtils.hpp"

#include "backend/cuda/gpuKernelDef.h"

namespace VideoStitch {
namespace Image {

struct BilinearLookupRGBA {
  typedef uint32_t Type;

  static inline __device__ Type outOfRangeValue() { return 0; }

  static inline __device__ void getSeperateChannels(const float2 uv, const Type topLeft, const Type topRight,
                                                    const Type bottomRight, const Type bottomLeft, int32_t& r,
                                                    int32_t& g, int32_t& b, uint32_t& a) {
    const int uTopLeft = floorf(uv.x);
    const int vTopLeft = floorf(uv.y);
    const float du = (uv.x - uTopLeft);
    const float dv = (uv.y - vTopLeft);
    a = ((topLeft & topRight & bottomLeft & bottomRight) >> 24) & (uint32_t)0xff;
    float fb = ((topLeft >> 16) & (uint32_t)0xff) * (1.0f - du) * (1.0f - dv) +
               ((topRight >> 16) & (uint32_t)0xff) * du * (1.0f - dv) +
               ((bottomLeft >> 16) & (uint32_t)0xff) * (1.0f - du) * dv +
               ((bottomRight >> 16) & (uint32_t)0xff) * du * dv;
    float fg = ((topLeft >> 8) & (uint32_t)0xff) * (1.0f - du) * (1.0f - dv) +
               ((topRight >> 8) & (uint32_t)0xff) * du * (1.0f - dv) +
               ((bottomLeft >> 8) & (uint32_t)0xff) * (1.0f - du) * dv +
               ((bottomRight >> 8) & (uint32_t)0xff) * du * dv;
    float fr = (topLeft & (uint32_t)0xff) * (1.0f - du) * (1.0f - dv) + (topRight & (uint32_t)0xff) * du * (1.0f - dv) +
               (bottomLeft & (uint32_t)0xff) * (1.0f - du) * dv + (bottomRight & (uint32_t)0xff) * du * dv;
    // discretize
    r = __float2int_rn(fr);
    g = __float2int_rn(fg);
    b = __float2int_rn(fb);
  }
};

struct BilinearLookupRGBAtoRGBA : public BilinearLookupRGBA {
  static inline __device__ Type interpolate(const float2 uv, const Type topLeft, const Type topRight,
                                            const Type bottomRight, const Type bottomLeft) {
    uint32_t a;
    int32_t r, g, b;
    getSeperateChannels(uv, topLeft, topRight, bottomRight, bottomLeft, r, g, b, a);
    return Image::RGBA::pack(Image::clamp8(r), Image::clamp8(g), Image::clamp8(b), a);
  }
};

template <typename BilinearLookup>
__device__ typename BilinearLookup::Type bilinearLookup(const float2 uv, const int2 size,
                                                        const typename BilinearLookup::Type* __restrict__ g_idata) {
  const int inputWidth = size.x;
  const int inputHeight = size.y;
  const int uTopLeft = floorf(uv.x);
  const int vTopLeft = floorf(uv.y);
  if (inRange(uv, size)) {
    const typename BilinearLookup::Type topLeft = g_idata[vTopLeft * inputWidth + uTopLeft];
    const typename BilinearLookup::Type topRight =
        uTopLeft + 1 < inputWidth ? g_idata[vTopLeft * inputWidth + uTopLeft + 1] : topLeft;
    const typename BilinearLookup::Type bottomRight = uTopLeft + 1 < inputWidth && vTopLeft + 1 < inputHeight
                                                          ? g_idata[(vTopLeft + 1) * inputWidth + uTopLeft + 1]
                                                          : topRight;
    const typename BilinearLookup::Type bottomLeft =
        vTopLeft + 1 < inputHeight ? g_idata[(vTopLeft + 1) * inputWidth + uTopLeft] : topLeft;
    return BilinearLookup::interpolate(uv, topLeft, topRight, bottomRight, bottomLeft);
  } else {
    return BilinearLookup::outOfRangeValue();
  }
}

template <typename BoundaryCondition, typename BilinearInterpolation>
__global__ void upsample22Kernel(typename BilinearInterpolation::Type* __restrict__ dst,
                                 const typename BilinearInterpolation::Type* __restrict__ src, unsigned dstWidth,
                                 unsigned dstHeight, unsigned srcWidth, unsigned srcHeight) {
  typedef typename BilinearInterpolation::Type T;

  const unsigned srcX = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned srcY = blockIdx.y * blockDim.y + threadIdx.y;

  // Load the data.
  __shared__ T sharedSrc[(16 + 2) * (16 + 2)];
  loadToSharedMemory<T, 1, 1, 1, 1, BoundaryCondition>(sharedSrc, blockDim.x, blockDim.y, src, srcWidth, srcHeight,
                                                       blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
  __syncthreads();

  const unsigned sharedWidth = blockDim.x + 2;
  const unsigned sharedSrcIdx = sharedWidth * (threadIdx.y + 1) + threadIdx.x + 1;
  if (srcX < srcWidth && srcY < srcHeight) {
    const unsigned dstX = 2 * srcX;
    const unsigned dstY = 2 * srcY;
    // +=======+=======+=======+
    // |       |       |       |
    // |   A   |   B   |   C   |
    // |       |       |       |
    // +=======+===+===+=======+
    // |       | a | b |       |
    // |   D   +---+---+   F   |
    // |       | c | d |       |
    // +=======+===+===+=======+
    // |       |       |       |
    // |   G   |   H   |   I   |
    // |       |       |       |
    // +=======+=======+=======+
    const T A = sharedSrc[sharedSrcIdx - sharedWidth - 1];
    const T B = sharedSrc[sharedSrcIdx - sharedWidth];
    const T C = sharedSrc[sharedSrcIdx - sharedWidth + 1];
    const T D = sharedSrc[sharedSrcIdx - 1];
    const T E = sharedSrc[sharedSrcIdx];
    const T F = sharedSrc[sharedSrcIdx + 1];
    const T G = sharedSrc[sharedSrcIdx + sharedWidth - 1];
    const T H = sharedSrc[sharedSrcIdx + sharedWidth];
    const T I = sharedSrc[sharedSrcIdx + sharedWidth + 1];
    // a
    if (dstX < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX] = BilinearInterpolation::interpolate(E, D, B, A);
    }
    // b
    if (dstX + 1 < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX + 1] = BilinearInterpolation::interpolate(E, F, B, C);
    }
    // c
    if (dstX < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX] = BilinearInterpolation::interpolate(E, D, H, G);
    }
    // d
    if (dstX + 1 < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX + 1] = BilinearInterpolation::interpolate(E, F, H, I);
    }
  }
}

}  // namespace Image
}  // namespace VideoStitch
