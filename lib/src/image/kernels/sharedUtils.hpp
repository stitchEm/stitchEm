// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SHARED_UTILS_HPP_
#define SHARED_UTILS_HPP_

// This part of the code is available only to CUDA.
#ifndef __CUDACC__
#error "You included this file in non-device code. This should not happen."
#endif

#include <stdio.h>
#include <math_constants.h>
namespace VideoStitch {
namespace Image {

/**
 * Fixed constant boundary condition.
 */
template <typename T>
struct ZeroBoundary {
  static inline __device__ T bottomRightValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                              const unsigned srcX, const unsigned srcY) {
    if (srcX < srcWidth && srcY < srcHeight) {
      return src[srcWidth * srcY + srcX];
    }
    return (T)0;
  }

  static inline __device__ T leftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight, const int srcX,
                                       const unsigned srcY) {
    if (srcX >= 0 && srcY < srcHeight) {
      return src[srcWidth * srcY + (unsigned)srcX];
    }
    return (T)0;
  }

  static inline __device__ T topValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                      const unsigned srcX, const int srcY) {
    if (srcX < srcWidth && srcY >= 0) {
      return src[srcWidth * (unsigned)srcY + srcX];
    }
    return (T)0;
  }

  static inline __device__ T topLeftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                          const int srcX, const int srcY) {
    if (srcX >= 0 && srcY >= 0) {
      return src[srcWidth * (unsigned)srcY + (unsigned)srcX];
    }
    return (T)0;
  }
};

struct MinInfBoundary {
  static inline __device__ float bottomRightValue(const float* src, const unsigned srcWidth, const unsigned srcHeight,
                                                  const unsigned srcX, const unsigned srcY) {
    if (srcX < srcWidth && srcY < srcHeight) {
      return src[srcWidth * srcY + srcX];
    }
    return CUDART_INF_F;
  }

  static inline __device__ float leftValue(const float* src, const unsigned srcWidth, const unsigned srcHeight,
                                           const int srcX, const unsigned srcY) {
    if (srcX >= 0 && srcY < srcHeight) {
      return src[srcWidth * srcY + (unsigned)srcX];
    }
    return CUDART_INF_F;
  }

  static inline __device__ float topValue(const float* src, const unsigned srcWidth, const unsigned srcHeight,
                                          const unsigned srcX, const int srcY) {
    if (srcX < srcWidth && srcY >= 0) {
      return src[srcWidth * (unsigned)srcY + srcX];
    }
    return CUDART_INF_F;
  }

  static inline __device__ float topLeftValue(const float* src, const unsigned srcWidth, const unsigned srcHeight,
                                              const int srcX, const int srcY) {
    if (srcX >= 0 && srcY >= 0) {
      return src[srcWidth * (unsigned)srcY + (unsigned)srcX];
    }
    return CUDART_INF_F;
  }
};

/**
 * Constant (extend) boundary condition.
 */
template <typename T>
struct ExtendBoundary {
  static inline __device__ T bottomRightValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                              const unsigned srcX, const unsigned srcY) {
    const unsigned x = srcX < srcWidth ? srcX : srcWidth - 1;
    const unsigned y = srcY < srcHeight ? srcY : srcHeight - 1;
    return src[srcWidth * y + x];
  }

  static inline __device__ T leftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight, const int srcX,
                                       const unsigned srcY) {
    const unsigned x = srcX >= 0 ? (unsigned)srcX : 0u;
    const unsigned y = srcY < srcHeight ? srcY : srcHeight - 1;
    return src[srcWidth * y + x];
  }

  static inline __device__ T topValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                      const unsigned srcX, const int srcY) {
    const unsigned x = srcX < srcWidth ? srcX : srcWidth - 1;
    const unsigned y = srcY >= 0 ? (unsigned)srcY : 0u;
    return src[srcWidth * y + x];
  }

  static inline __device__ T topLeftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                          const int srcX, const int srcY) {
    const unsigned x = srcX >= 0 ? (unsigned)srcX : 0u;
    const unsigned y = srcY >= 0 ? (unsigned)srcY : 0u;
    return src[srcWidth * y + x];
  }
};

/**
 * Wrapping boundary condition.
 */
template <typename T>
struct WrapBoundary {
  static inline __device__ T bottomRightValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                              const unsigned srcX, const unsigned srcY) {
    const unsigned x = srcX % srcWidth;
    const unsigned y = srcY % srcHeight;
    return src[srcWidth * y + x];
  }

  static inline __device__ T leftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight, const int srcX,
                                       const unsigned srcY) {
    const unsigned x = (unsigned)(srcX >= 0 ? srcX : (int)srcWidth + srcX);
    const unsigned y = srcY % srcHeight;
    return src[srcWidth * y + x];
  }

  static inline __device__ T topValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                      const unsigned srcX, const int srcY) {
    const unsigned x = srcX % srcWidth;
    const unsigned y = srcY >= 0 ? (unsigned)srcY : 0u;
    return src[srcWidth * y + x];
  }

  static inline __device__ T topLeftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                          const int srcX, const int srcY) {
    const unsigned x = (unsigned)(srcX >= 0 ? srcX : (int)srcWidth + srcX);
    const unsigned y = srcY >= 0 ? (unsigned)srcY : 0u;
    return src[srcWidth * y + x];
  }
};

/**
 * Horizontal wrapping boundary condition.
 */
template <typename T>
struct HWrapBoundary {
  static inline __device__ T bottomRightValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                              const unsigned srcX, const unsigned srcY) {
    const unsigned x = srcX % srcWidth;
    const unsigned y = srcY < srcHeight ? srcY : srcHeight - 1;
    return src[srcWidth * y + x];
  }

  static inline __device__ T leftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight, const int srcX,
                                       const unsigned srcY) {
    const unsigned x = (unsigned)(srcX >= 0 ? srcX : (int)srcWidth + srcX);
    const unsigned y = srcY < srcHeight ? srcY : srcHeight - 1;
    return src[srcWidth * y + x];
  }

  static inline __device__ T topValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                      const unsigned srcX, const int srcY) {
    const unsigned x = srcX % srcWidth;
    const unsigned y = srcY >= 0 ? (unsigned)srcY : 0u;
    return src[srcWidth * y + x];
  }

  static inline __device__ T topLeftValue(const T* src, const unsigned srcWidth, const unsigned srcHeight,
                                          const int srcX, const int srcY) {
    const unsigned x = (unsigned)(srcX >= 0 ? srcX : (int)srcWidth + srcX);
    const unsigned y = srcY >= 0 ? (unsigned)srcY : 0u;
    return src[srcWidth * y + x];
  }
};

/**
 * Load a part of @a src to @a sharedDst. In addition to the core pixels, we will load @a left additional pixels to the
 * left (same for @a right, @a top, @a bottom). Outside of the source, pixels values are taken to be the same as the
 * border pixels. Pixels of @a sharedDst further than @a left (resp @a right, @a top, @a bottom) of any of the borders
 * of src have an undefined value.
 * @param sharedDst destination buffer, of size (@a sharedWidth + @a moreLeft + @a moreRight) x (@a sharedHeight + @a
 * moreTop + @a moreBottom)
 * @param sharedWidth base width of the shared array
 * @param sharedHeight base height of the shared array
 * @param src source buffer
 * @param srcWidth width of the source buffer.
 * @param srcHeight width of the source buffer
 *
 * @a Getter defines how to retrieve values outside of boundaries. See above for options.
 *
 * WARNING: you need ot call __syncthreads() before reading the shared memory.
 * It's not done in this function so that you can do something else in between.
 *
 * TODO: version that's templated on shared width size for loop unrolling.
 */
template <typename T, unsigned left, unsigned right, unsigned top, unsigned bottom, typename Getter>
inline __device__ void loadToSharedMemory(T* __restrict__ sharedDst, const unsigned sharedWidth,
                                          const unsigned sharedHeight, const T* __restrict__ src,
                                          const unsigned srcWidth, const unsigned srcHeight, const unsigned srcOffsetX,
                                          const unsigned srcOffsetY) {
  const unsigned threadId = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned realSharedWidth = sharedWidth + left + right;

  // Start with interior pixels.
  for (int i = threadId; i < sharedWidth * sharedHeight; i += blockDim.x * blockDim.y) {
    const unsigned sharedX = i % sharedWidth;
    const unsigned sharedY = i / sharedWidth;
    sharedDst[realSharedWidth * top + left + realSharedWidth * sharedY + sharedX] =
        Getter::bottomRightValue(src, srcWidth, srcHeight, srcOffsetX + sharedX, srcOffsetY + sharedY);
  }

  // Top interior pixels
  for (int i = threadId; i < sharedWidth * top; i += blockDim.x * blockDim.y) {
    const unsigned sharedX = i % sharedWidth;
    const unsigned sharedY = i / sharedWidth;
    sharedDst[left + realSharedWidth * sharedY + sharedX] =
        Getter::topValue(src, srcWidth, srcHeight, srcOffsetX + sharedX, (int)srcOffsetY + (int)sharedY - (int)top);
  }

  // Left interior pixels
  for (int i = threadId; i < sharedHeight * left; i += blockDim.x * blockDim.y) {
    const unsigned sharedX = i % left;
    const unsigned sharedY = i / left;
    sharedDst[realSharedWidth * top + realSharedWidth * sharedY + sharedX] =
        Getter::leftValue(src, srcWidth, srcHeight, (int)srcOffsetX + (int)sharedX - (int)left, srcOffsetY + sharedY);
  }

  // Bottom interior pixels
  for (int i = threadId; i < sharedWidth * bottom; i += blockDim.x * blockDim.y) {
    const unsigned sharedX = i % sharedWidth;
    const unsigned sharedY = sharedHeight + i / sharedWidth;
    sharedDst[realSharedWidth * top + left + realSharedWidth * sharedY + sharedX] =
        Getter::bottomRightValue(src, srcWidth, srcHeight, srcOffsetX + sharedX, srcOffsetY + sharedY);
  }

  // Right interior pixels
  for (int i = threadId; i < sharedHeight * right; i += blockDim.x * blockDim.y) {
    const unsigned sharedX = sharedWidth + i % right;
    const unsigned sharedY = i / right;
    sharedDst[realSharedWidth * top + left + realSharedWidth * sharedY + sharedX] =
        Getter::bottomRightValue(src, srcWidth, srcHeight, srcOffsetX + sharedX, srcOffsetY + sharedY);
  }

  // NOTE: Hereafter we assume that the block size is enough to load all corner pixels.

  // Top-left pixels
  if (threadId < left * top) {
    // nvcc does not understand that the condition above makes it impossible for 'left' to be zero.
    const unsigned sharedX = threadId % left;
    const unsigned sharedY = threadId / left;
    sharedDst[realSharedWidth * sharedY + sharedX] =
        Getter::topLeftValue(src, srcWidth, srcHeight, (int)srcOffsetX + (int)sharedX - (int)left,
                             (int)srcOffsetY + (int)sharedY - (int)top);
  }

  // Top-right pixels
  if (threadId < right * top) {
    // nvcc does not understand that the condition above makes it impossible for 'right' to be zero.
    const unsigned sharedX = sharedWidth + threadId % right;
    const unsigned sharedY = threadId / right;
    sharedDst[left + realSharedWidth * sharedY + sharedX] =
        Getter::topValue(src, srcWidth, srcHeight, srcOffsetX + sharedX, (int)srcOffsetY + (int)sharedY - (int)top);
  }

  // Bottom-left pixels
  if (threadId < left * bottom) {
    // nvcc does not understand that the condition above makes it impossible for 'left' to be zero.
    const unsigned sharedX = threadId % left;
    const unsigned sharedY = sharedHeight + threadId / left;
    sharedDst[realSharedWidth * top + realSharedWidth * sharedY + sharedX] =
        Getter::leftValue(src, srcWidth, srcHeight, (int)srcOffsetX + (int)sharedX - (int)left, srcOffsetY + sharedY);
  }

  // Bottom-right pixels
  if (threadId < right * bottom) {
    // nvcc does not understand that the condition above makes it impossible for 'right' to be zero.
    const unsigned sharedX = sharedWidth + threadId % right;
    const unsigned sharedY = sharedHeight + threadId / right;
    sharedDst[left + realSharedWidth * top + realSharedWidth * sharedY + sharedX] =
        Getter::bottomRightValue(src, srcWidth, srcHeight, srcOffsetX + sharedX, srcOffsetY + sharedY);
  }
}

}  // namespace Image
}  // namespace VideoStitch
#endif
