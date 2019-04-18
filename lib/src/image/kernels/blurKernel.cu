// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// CUDA implementation of box blur.
// Slightly modified from NVIDIA's SDK samples.

#ifndef _BLURKERNEL_H_
#define _BLURKERNEL_H_

#include "gpu/image/blur.hpp"

#include "backend/common/imageOps.hpp"

#include "backend/common/image/blurdef.h"

namespace VideoStitch {
namespace Image {
/**
 * Accumulator values for several types.
 */
template <typename U>
struct AccumT {};

/**
 * Accumulator values for float.
 */
template <>
struct AccumT<float> {
  typedef float Type;
  typedef float DividerType;
  static __device__ float init(const float s) { return s; }
};

/**
 * Accumulator values for float2.
 */
template <>
struct AccumT<float2> {
  typedef float2 Type;
  typedef float2 DividerType;
  static __device__ float2 init(const float s) { return make_float2(s, s); }
};

/**
 * Accumulator values for uchar.
 */
template <>
struct AccumT<unsigned char> {
  typedef unsigned Type;
  typedef unsigned DividerType;
  static __device__ unsigned init(const unsigned s) { return s; }
};

/**
 * A class that accumulates values of type T.
 * The default implementation is for scalar values only.
 */
template <typename T>
class Accumulator {
 public:
  __device__ Accumulator(int radius) : acc(AccumT<T>::init(0)), divider(AccumT<T>::init(2 * radius + 1)) {}

  /**
   * Accumulates a value.
   * @param v
   */
  __device__ void accumulate(const T v) { acc += v; }

  /**
   * Unaccumulates a value.
   * @param v
   */
  __device__ void unaccumulate(const T v) { acc -= v; }

  /**
   * Returns the divided accumulated (blurred) pixel value.
   * Parameters are unused but are here to provide the same API as Accumulator<uint32_t>.
   */
  __device__ T get(const T* /*src*/, size_t /*i*/) const {
    return (divider == AccumT<T>::init(0)) ? AccumT<T>::init(0) : (acc / divider);
  }

 private:
  typename AccumT<T>::Type acc;
  const typename AccumT<T>::DividerType divider;
};

/** Maybe I'm overlooking something, but I don't understand why nvcc gives warning about all 4 private fields being
 * unused. They are clearly used, or this Accumulator would not accumulate anything.
 */

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

/**
 * Accumulator for RGBA210 values. Pixels with 0 alpha contribute 0.
 * So the divider will always be between 0 and 2 * radius + 1.
 */
template <>
class Accumulator<uint32_t> {
 public:
  __device__ Accumulator(int /*radius*/) : accR(0), accG(0), accB(0), divider(0) {}

  /**
   * Accumulates a value.
   * @param v
   */
  __device__ void accumulate(const uint32_t v) {
    const int32_t isSolid = RGB210::a(v);
    if (isSolid != 0) {
      accR += RGB210::r(v);
      accG += RGB210::g(v);
      accB += RGB210::b(v);
      ++divider;
    }
  }

  /**
   * Unaccumulates a value.
   * @param v
   */
  __device__ void unaccumulate(const uint32_t v) {
    const int32_t isSolid = RGB210::a(v);
    if (isSolid != 0) {
      accR -= RGB210::r(v);
      accG -= RGB210::g(v);
      accB -= RGB210::b(v);
      --divider;
    }
  }

  /**
   * Returns the divided accumulated (blurred) pixel value.
   */
  __device__ uint32_t get(const uint32_t* src, size_t i) const {
    return (divider == 0) ? 0 : RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(src[i]));
  }

 private:
  int32_t accR;
  int32_t accG;
  int32_t accB;
  int32_t divider;
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

/**
 * Gaussian blur in 1D. @a r is the filter radius, which must be such that 2 * r < h.
 * NOTE: STRONG rounding artifacts with T==integer type
 */
template <typename T>
__global__ void blur1DKernelNoWrap(T* __restrict__ dst, const T* __restrict__ src, int w, int h, const int r) {
  int columnId = blockIdx.x * blockDim.x + threadIdx.x;

  if (columnId < w) {
    dst += columnId;
    src += columnId;
    Accumulator<T> accumulator(r);

    // Boundary condition: extend.
    const T v0 = src[0];
    accumulator.accumulate(v0);
    for (int y = 1; y < (r + 1); ++y) {
      accumulator.accumulate(v0);
      accumulator.accumulate(src[y * w]);
    }
    dst[0] = accumulator.get(src, 0);
    for (int y = 1; y < (r + 1); ++y) {
      accumulator.accumulate(src[(y + r) * w]);
      accumulator.unaccumulate(v0);
      dst[y * w] = accumulator.get(src, y * w);
    }

    // Main loop
    for (int y = (r + 1); y < (h - r); ++y) {
      accumulator.accumulate(src[(y + r) * w]);
      accumulator.unaccumulate(src[((y - r) * w) - w]);
      dst[y * w] = accumulator.get(src, y * w);
    }

    // Boundary condition: extend.
    const T vEnd = src[(h - 1) * w];
    for (int y = h - r; y < h; ++y) {
      accumulator.accumulate(vEnd);
      accumulator.unaccumulate(src[((y - r) * w) - w]);
      dst[y * w] = accumulator.get(src, y * w);
    }
  }
}

/**
 * Gaussian blur in 1D for cases where 2 * r >= h. r must be such that: r < h
 */
template <typename T>
__global__ void blur1DKernelNoWrapLargeRadius(T* __restrict__ dst, const T* __restrict__ src, int w, int h,
                                              const int r) {
  int columnId = blockIdx.x * blockDim.x + threadIdx.x;

  if (columnId < w) {
    dst += columnId;
    src += columnId;
    Accumulator<T> accumulator(r);

    // Boundary condition: extend.
    const T v0 = src[0];
    accumulator.accumulate(v0);
    for (int y = 1; y < (r + 1); ++y) {
      accumulator.accumulate(v0);
      accumulator.accumulate(src[y * w]);
    }
    dst[0] = accumulator.get(src, 0);
    // Stops at (h - r - 1) instead of (r + 1).
    for (int y = 1; y < (h - r); ++y) {
      accumulator.accumulate(src[(y + r) * w]);
      accumulator.unaccumulate(v0);
      dst[y * w] = accumulator.get(src, y * w);
    }

    const T vEnd = src[(h - 1) * w];
    // Middle loop
    for (int y = h - r; y < (r + 1); ++y) {
      accumulator.accumulate(vEnd);
      accumulator.unaccumulate(v0);
      dst[y * w] = accumulator.get(src, y * w);
    }

    // Boundary condition: extend.
    for (int y = r + 1; y < h; ++y) {
      accumulator.accumulate(vEnd);
      accumulator.unaccumulate(src[((y - r) * w) - w]);
      dst[y * w] = accumulator.get(src, y * w);
    }
  }
}

/**
 * Gaussian blur in 1D for cases where r >= h. Here only the boundary conditions apply since all the buffer values are
 * in the stencil, always.
 */
template <typename T>
__global__ void blur1DKernelNoWrapHugeRadius(T* __restrict__ dst, const T* __restrict__ src, int w, int h,
                                             const int r) {
  int columnId = blockIdx.x * blockDim.x + threadIdx.x;

  if (columnId < w) {
    dst += columnId;
    src += columnId;
    Accumulator<T> accumulator(r);

    // Boundary condition: extend.
    const T v0 = src[0];
    const T vEnd = src[(h - 1) * w];
    for (int y = 0; y < r; ++y) {
      accumulator.accumulate(v0);
    }
    // Accumulate all buffer values.
    for (int y = 0; y < h; ++y) {
      accumulator.accumulate(src[y * w]);
    }
    // Fill up with past-end-of-buffer values.
    for (int y = h; y < r + 1; ++y) {
      accumulator.accumulate(vEnd);
    }

    // Then everything is simple.
    dst[0] = accumulator.get(src, 0);
    for (int y = 1; y < h; ++y) {
      accumulator.accumulate(vEnd);
      accumulator.unaccumulate(v0);
      dst[y * w] = accumulator.get(src, y * w);
    }
  }
}

/**
 * Same, but wraps
 */
template <typename T>
__global__ void blur1DKernelWrap(T* __restrict__ dst, const T* __restrict__ src, int w, int h, const int r) {
  int columnId = blockIdx.x * blockDim.x + threadIdx.x;

  if (columnId < w) {
    dst += columnId;
    src += columnId;
    Accumulator<T> accumulator(r);

    // Boundary condition: wrap.
    for (int y = h - r; y < h; ++y) {
      accumulator.accumulate(src[y * w]);
    }
    for (int y = 0; y < (r + 1); ++y) {
      accumulator.accumulate(src[y * w]);
    }
    dst[0] = accumulator.get(src, 0);
    for (int y = 1; y < (r + 1); ++y) {
      accumulator.accumulate(src[(y + r) * w]);
      accumulator.unaccumulate(src[(h + y - r) * w - w]);
      dst[y * w] = accumulator.get(src, y * w);
    }

    // Main loop
    for (int y = (r + 1); y < (h - r); ++y) {
      accumulator.accumulate(src[(y + r) * w]);
      accumulator.unaccumulate(src[(y - r) * w - w]);
      dst[y * w] = accumulator.get(src, y * w);
    }

    // Boundary condition: wrap.
    for (int y = h - r; y < h; ++y) {
      accumulator.accumulate(src[(y + r - h) * w]);
      accumulator.unaccumulate(src[((y - r) * w) - w]);
      dst[y * w] = accumulator.get(src, y * w);
    }
  }
}

/*
 * bluColumnsKernel is a modification of convolution separable kernel from Nvidia samples.
 * It adds to the use of shared memory the accumulation algorithm.
 * Each thread blurs COLUMNS_RESULT_STEPS consecutive pixels on the Y dimension.
 */

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

template <typename T>
__global__ void blurColumnsKernelNoWrap(T* dst, const T* src, int width, int height, int pitch, int radius) {
  const int idx = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;  // thread id on x dimension
  const int idy = blockIdx.y * COLUMNS_BLOCKDIM_Y + threadIdx.y;  // thread id on y dimension
  if (idx < width) {                                              // check if thread is not out of the image

    // Shared buffer is a 2D array represented as a 1D array. Each thread blurs COLUMNS_RESULT_STEPS pixels on the Y
    // dimension.

    __shared__ T s_Data[COLUMNS_BLOCKDIM_X * SIZE_Y];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

    const T* InputWithHaloOffset = src + baseY * pitch + baseX;  // move for reading
    T* OutputWithOffset =
        dst + (threadIdx.y * COLUMNS_RESULT_STEPS + (blockIdx.y * COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)) * pitch +
        baseX;  // move for writing

    // load data needed by the block into shared memory
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; ++i) {
      if ((baseY >= (-i * COLUMNS_BLOCKDIM_Y)) && (height - baseY > i * COLUMNS_BLOCKDIM_Y)) {  // inside image
        s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            InputWithHaloOffset[i * COLUMNS_BLOCKDIM_Y * pitch];
      } else {
        if (baseY < -i * COLUMNS_BLOCKDIM_Y) {  // out of image (upper)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = InputWithHaloOffset[-baseY * pitch];

        } else {  // out of image (lower)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
              InputWithHaloOffset[(height - 1 - baseY) * pitch];
        }
      }
    }
    __syncthreads();
    Accumulator<T> acc(radius);

    // every thread blurs COLUMNS_RESULT_STEPS pixels starting from this offset (skipping the halo)
    const int offset =
        threadIdx.x * SIZE_Y + COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS + threadIdx.y * COLUMNS_RESULT_STEPS;

    if (idy * COLUMNS_RESULT_STEPS < height) {  // check if thread is not out of the image
#pragma unroll
      for (int j = -radius; j <= radius; ++j) {
        T v = s_Data[offset + j];
        acc.accumulate(v);
      }
      OutputWithOffset[0] = acc.get(s_Data, offset);

      // every thread blurs COLUMNS_RESULT_STEPS pixels, unless it is in the very low part of the image

      for (int i = 1; i < MIN(COLUMNS_RESULT_STEPS, height - idy * COLUMNS_RESULT_STEPS); ++i) {
        T v0 = s_Data[offset + i + radius];
        acc.accumulate(v0);

        T v1 = s_Data[offset + i - radius - 1];
        acc.unaccumulate(v1);

        OutputWithOffset[i * pitch] = acc.get(s_Data, offset + i);
      }
    }
  }
}

template <>
__global__ void blurColumnsKernelNoWrap<uint32_t>(uint32_t* dst, const uint32_t* src, int width, int height, int pitch,
                                                  int radius) {
  const int idx = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;  // thread id on x dimension
  const int idy = blockIdx.y * COLUMNS_BLOCKDIM_Y + threadIdx.y;  // thread id on y dimension

  if (idx < width) {  // check if thread is not out of the image

    // Shared buffer is a 2D array represented as a 1D array. Each thread blurs COLUMNS_RESULT_STEPS pixels on the Y
    // dimension.

    __shared__ uint32_t s_Data[COLUMNS_BLOCKDIM_X * SIZE_Y];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

    const uint32_t* InputWithHaloOffset = src + baseY * pitch + baseX;  // move for reading
    uint32_t* OutputWithOffset =
        dst + (threadIdx.y * COLUMNS_RESULT_STEPS + (blockIdx.y * COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)) * pitch +
        baseX;  // move for writing

    // load data needed by the block into shared memory
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; ++i) {
      if ((baseY >= (-i * COLUMNS_BLOCKDIM_Y)) && (height - baseY > i * COLUMNS_BLOCKDIM_Y)) {  // inside image
        s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            InputWithHaloOffset[i * COLUMNS_BLOCKDIM_Y * pitch];
      } else {
        if (baseY < -i * COLUMNS_BLOCKDIM_Y) {  // out of image (upper)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = InputWithHaloOffset[-baseY * pitch];

        } else {  // out of image (lower)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
              InputWithHaloOffset[(height - 1 - baseY) * pitch];
        }
      }
    }
    __syncthreads();

    // the use of accumlator class leads sometimes to a bug on linux which is unexplained. That's why we avoid it
    int32_t accR(0);
    int32_t accG(0);
    int32_t accB(0);
    int32_t divider(0);

    // every thread blurs  COLUMNS_RESULT_STEPS pixels starting from this offset (skipping the halo)
    const int offset =
        threadIdx.x * SIZE_Y + COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS + threadIdx.y * COLUMNS_RESULT_STEPS;

    if (idy * COLUMNS_RESULT_STEPS < height) {
      for (int j = -radius; j <= radius; ++j) {
        uint32_t v = s_Data[offset + j];
        if (RGB210::a(v) != 0) {
          accR += RGB210::r(v);
          accG += RGB210::g(v);
          accB += RGB210::b(v);
          divider++;
        }
      }
      if (divider != 0) {
        OutputWithOffset[0] = RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(s_Data[offset]));
      } else {
        OutputWithOffset[0] = 0;
      }

      // every thread blurs COLUMNS_RESULT_STEPS pixels, unless it is in the very low part of the image

      for (int i = 1; i < MIN(COLUMNS_RESULT_STEPS, height - idy * COLUMNS_RESULT_STEPS); ++i) {
        uint32_t v0 = s_Data[offset + i + radius];
        if (RGB210::a(v0) != 0) {
          accR += RGB210::r(v0);
          accG += RGB210::g(v0);
          accB += RGB210::b(v0);
          ++divider;
        }
        uint32_t v1 = s_Data[offset + i - radius - 1];

        if (RGB210::a(v1) != 0) {
          accR -= RGB210::r(v1);
          accG -= RGB210::g(v1);
          accB -= RGB210::b(v1);
          --divider;
        }

        if (divider != 0) {
          OutputWithOffset[i * pitch] =
              RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(s_Data[offset + i]));
        } else {
          OutputWithOffset[i * pitch] = 0;
        }
      }
    }
  }
}

template <typename T>
__global__ void blurColumnsKernelWrap(T* dst, const T* src, int width, int height, int pitch, int radius) {
  const int idx = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;  // thread id on x dimension
  const int idy = blockIdx.y * COLUMNS_BLOCKDIM_Y + threadIdx.y;  // thread id on y dimension
  if (idx < width) {                                              // check if thread is not out of the image

    // Shared buffer is a 2D array represented as a 1D array. Each thread blurs COLUMNS_RESULT_STEPS pixels on the Y
    // dimension.

    __shared__ T s_Data[COLUMNS_BLOCKDIM_X * SIZE_Y];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

    const T* InputWithHaloOffset = src + baseY * pitch + baseX;
    T* OutputWithOffset =
        dst + (threadIdx.y * COLUMNS_RESULT_STEPS + (blockIdx.y * COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)) * pitch +
        baseX;  // move for writing

    // load data needed by the block into shared memory
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
      if ((baseY >= (-i * COLUMNS_BLOCKDIM_Y)) && (height - baseY > i * COLUMNS_BLOCKDIM_Y)) {
        s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            InputWithHaloOffset[i * COLUMNS_BLOCKDIM_Y * pitch];
      } else {
        if (baseY < -i * COLUMNS_BLOCKDIM_Y) {  // out of image (upper)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
              InputWithHaloOffset[(height + i * COLUMNS_BLOCKDIM_Y) * pitch];
        } else {  // out of image (lower)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
              InputWithHaloOffset[(i * COLUMNS_BLOCKDIM_Y - height) * pitch];
        }
      }
    }
    __syncthreads();
    Accumulator<T> acc(radius);
    // every thread blurs  COLUMNS_RESULT_STEPS pixels starting from this offset (skipping the halo)
    const int offset =
        threadIdx.x * SIZE_Y + COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS + threadIdx.y * COLUMNS_RESULT_STEPS;
    if (idy * COLUMNS_RESULT_STEPS < height) {  // check if thread is not out of the image
#pragma unroll
      for (int j = -radius; j <= radius; j++) {
        T v = s_Data[offset + j];
        acc.accumulate(v);
      }
      OutputWithOffset[0] = acc.get(s_Data, offset);

      // every thread blurs COLUMNS_RESULT_STEPS pixels, unless it is in the very low part of the image

      for (int i = 1; i < MIN(COLUMNS_RESULT_STEPS, height - idy * COLUMNS_RESULT_STEPS); i++) {
        T v0 = s_Data[offset + i + radius];

        acc.accumulate(v0);
        T v1 = s_Data[offset + i - radius - 1];

        acc.unaccumulate(v1);
        OutputWithOffset[i * pitch] = acc.get(s_Data, offset + i);
      }
    }
  }
}

template <>
__global__ void blurColumnsKernelWrap<uint32_t>(uint32_t* dst, const uint32_t* src, int width, int height, int pitch,
                                                int radius) {
  const int idx = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;  // thread id on x dimension
  const int idy = blockIdx.y * COLUMNS_BLOCKDIM_Y + threadIdx.y;  // thread id on y dimension
  if (idx < width) {                                              // check if thread is not out of the image

    // Shared buffer is a 2D array represented as a 1D array. Each thread blurs COLUMNS_RESULT_STEPS pixels on the Y
    // dimension.
    __shared__ uint32_t s_Data[COLUMNS_BLOCKDIM_X * SIZE_Y];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

    const uint32_t* InputWithHaloOffset = src + baseY * pitch + baseX;
    uint32_t* OutputWithOffset =
        dst + (threadIdx.y * COLUMNS_RESULT_STEPS + (blockIdx.y * COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)) * pitch +
        baseX;  // moving for writing

    // load data needed by the block into shared memory
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
      if ((baseY >= (-i * COLUMNS_BLOCKDIM_Y)) && (height - baseY > i * COLUMNS_BLOCKDIM_Y)) {
        s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            InputWithHaloOffset[i * COLUMNS_BLOCKDIM_Y * pitch];
      } else {
        if (baseY < -i * COLUMNS_BLOCKDIM_Y) {  // out of image (upper)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
              InputWithHaloOffset[(height + i * COLUMNS_BLOCKDIM_Y) * pitch];
        } else {  // out of image (lower)
          s_Data[threadIdx.x * SIZE_Y + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
              InputWithHaloOffset[(i * COLUMNS_BLOCKDIM_Y - height) * pitch];
        }
      }
    }
    __syncthreads();

    // the use of accumlator class leads sometimes to a bug on linux which is unexplained. That's why we avoid it
    int32_t accR(0);
    int32_t accG(0);
    int32_t accB(0);
    int32_t divider(0);

    // every thread blurs  COLUMNS_RESULT_STEPS pixels starting from this offset (skipping the halo)
    const int offset =
        threadIdx.x * SIZE_Y + COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS + threadIdx.y * COLUMNS_RESULT_STEPS;

    if (idy * COLUMNS_RESULT_STEPS < height) {
      for (int j = -radius; j <= radius; j++) {
        uint32_t v = s_Data[offset + j];
        if (RGB210::a(v) != 0) {
          accR += RGB210::r(v);
          accG += RGB210::g(v);
          accB += RGB210::b(v);
          divider++;
        }
      }
      if (divider != 0) {
        OutputWithOffset[0] = RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(s_Data[offset]));
      } else {
        OutputWithOffset[0] = 0;
      }

      // every thread blurs COLUMNS_RESULT_STEPS pixels, unless it is in the very low part of the image
      for (int i = 1; i < MIN(COLUMNS_RESULT_STEPS, height - idy * COLUMNS_RESULT_STEPS); i++) {
        uint32_t v0 = s_Data[offset + i + radius];
        if (RGB210::a(v0) != 0) {
          accR += RGB210::r(v0);
          accG += RGB210::g(v0);
          accB += RGB210::b(v0);
          ++divider;
        }
        uint32_t v1 = s_Data[offset + i - radius - 1];

        if (RGB210::a(v1) != 0) {
          accR -= RGB210::r(v1);
          accG -= RGB210::g(v1);
          accB -= RGB210::b(v1);
          --divider;
        }

        if (divider != 0) {
          OutputWithOffset[i * pitch] =
              RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(s_Data[offset + i]));
        } else {
          OutputWithOffset[i * pitch] = 0;
        }
      }
    }
  }
}

__global__ void blurRowsKernelNoWrap(uint32_t* dst, const uint32_t* src, std::size_t width, std::size_t height,
                                     std::size_t pitch, int radius) {
  __shared__ uint32_t s_Data_Input[ROWS_BLOCKDIM_Y][SIZE_X];
  __shared__ uint32_t s_Data_Output[ROWS_BLOCKDIM_Y][SIZE_X];

  // Offset to the left halo edge
  const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  const uint32_t* InputWithHaloOffset = src + baseY * pitch + baseX;
  uint32_t* OutputWithHaloOffset = dst + baseY * pitch + baseX;

  const int idy = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
  const int idx = blockIdx.x * ROWS_BLOCKDIM_X + threadIdx.x;

  if (idy < height) {
#pragma unroll
    // load data needed by the block into shared memory
    for (int i = 0; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
      if (((width - baseX) > (i * ROWS_BLOCKDIM_X)) && (baseX >= -i * ROWS_BLOCKDIM_X)) {
        s_Data_Input[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = InputWithHaloOffset[i * ROWS_BLOCKDIM_X];
      } else {
        if (baseX < -i * ROWS_BLOCKDIM_X) {  // out of image (left)
          s_Data_Input[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = InputWithHaloOffset[-baseX];
        } else {  // out of image (right)
          s_Data_Input[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = InputWithHaloOffset[width - baseX - 1];
        }
      }
    }
    // Compute and store results
    __syncthreads();
    int32_t accR(0);
    int32_t accG(0);
    int32_t accB(0);
    int32_t divider(0);

    // every thread blurs  ROWS_RESULT_STEPS pixels starting from this offset
    const int offset = ROWS_HALO_STEPS * ROWS_BLOCKDIM_X + threadIdx.x * ROWS_RESULT_STEPS;

    if (idx * ROWS_RESULT_STEPS < width) {
      for (int j = -radius; j <= radius; j++) {
        uint32_t v = s_Data_Input[threadIdx.y][offset + j];
        if (RGB210::a(v) != 0) {
          accR += RGB210::r(v);
          accG += RGB210::g(v);
          accB += RGB210::b(v);
          ++divider;
        }
      }
      if (divider != 0) {
        s_Data_Output[threadIdx.y][offset] =
            RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(s_Data_Input[threadIdx.y][offset]));
      } else {
        s_Data_Output[threadIdx.y][offset] = 0;
      }

      for (int i = 1; i < MIN(ROWS_RESULT_STEPS, width - idx * ROWS_RESULT_STEPS); i++) {
        uint32_t v0 = s_Data_Input[threadIdx.y][offset + i + radius];
        if (RGB210::a(v0) != 0) {
          accR += RGB210::r(v0);
          accG += RGB210::g(v0);
          accB += RGB210::b(v0);
          ++divider;
        }

        uint32_t v1 = s_Data_Input[threadIdx.y][offset + i - radius - 1];
        if (RGB210::a(v1) != 0) {
          accR -= RGB210::r(v1);
          accG -= RGB210::g(v1);
          accB -= RGB210::b(v1);
          --divider;
        }

        if (divider != 0) {
          s_Data_Output[threadIdx.y][offset + i] = RGB210::pack(accR / divider, accG / divider, accB / divider,
                                                                RGB210::a(s_Data_Input[threadIdx.y][offset + i]));
        } else {
          s_Data_Output[threadIdx.y][offset + i] = 0;
        }
      }
    }
    __syncthreads();

    // write to global memory (coalesced access)
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
      if ((i * ROWS_BLOCKDIM_X + baseX) < width) {
        OutputWithHaloOffset[i * ROWS_BLOCKDIM_X] = s_Data_Output[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X];
      }
    }
  }
}

__global__ void blurRowsKernelWrap(uint32_t* dst, const uint32_t* src, std::size_t width, std::size_t height,
                                   std::size_t pitch, int radius) {
  __shared__ uint32_t s_Data_Input[ROWS_BLOCKDIM_Y][SIZE_X];
  __shared__ uint32_t s_Data_Output[ROWS_BLOCKDIM_Y][SIZE_X];

  // Offset to the left halo edge
  const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  const uint32_t* InputWithHaloOffset = src + baseY * pitch + baseX;
  uint32_t* OutputWithHaloOffset = dst + baseY * pitch + baseX;

  const int idy = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
  const int idx = blockIdx.x * ROWS_BLOCKDIM_X + threadIdx.x;

  if (idy < height) {
#pragma unroll
    // load data needed by the block into shared memory
    for (int i = 0; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
      if (((width - baseX) > (i * ROWS_BLOCKDIM_X)) && (baseX >= -i * ROWS_BLOCKDIM_X)) {
        s_Data_Input[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = InputWithHaloOffset[i * ROWS_BLOCKDIM_X];
      } else {
        if (baseX < -i * ROWS_BLOCKDIM_X) {  // out of image (left)
          s_Data_Input[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
              InputWithHaloOffset[width + i * ROWS_BLOCKDIM_X];
        } else {  // out of image (right)
          s_Data_Input[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
              InputWithHaloOffset[i * ROWS_BLOCKDIM_X - width];
        }
      }
    }
    __syncthreads();
    // Compute and store results
    int32_t accR(0);
    int32_t accG(0);
    int32_t accB(0);
    int32_t divider(0);

    // every thread blurs  ROWS_RESULT_STEPS consecutive pixels starting from this offset
    const int offset = ROWS_HALO_STEPS * ROWS_BLOCKDIM_X + threadIdx.x * ROWS_RESULT_STEPS;

    if (idx * ROWS_RESULT_STEPS < width) {
      for (int j = -radius; j <= radius; j++) {
        uint32_t v = s_Data_Input[threadIdx.y][offset + j];
        if (RGB210::a(v) != 0) {
          accR += RGB210::r(v);
          accG += RGB210::g(v);
          accB += RGB210::b(v);
          ++divider;
        }
      }
      if (divider != 0) {
        s_Data_Output[threadIdx.y][offset] =
            RGB210::pack(accR / divider, accG / divider, accB / divider, RGB210::a(s_Data_Input[threadIdx.y][offset]));
      } else {
        s_Data_Output[threadIdx.y][offset] = 0;
      }

      for (int i = 1; i < MIN(ROWS_RESULT_STEPS, width - idx * ROWS_RESULT_STEPS); i++) {
        uint32_t v0 = s_Data_Input[threadIdx.y][offset + i + radius];
        if (RGB210::a(v0) != 0) {
          accR += RGB210::r(v0);
          accG += RGB210::g(v0);
          accB += RGB210::b(v0);
          ++divider;
        }

        uint32_t v1 = s_Data_Input[threadIdx.y][offset + i - radius - 1];
        if (RGB210::a(v1) != 0) {
          accR -= RGB210::r(v1);
          accG -= RGB210::g(v1);
          accB -= RGB210::b(v1);
          --divider;
        }

        if (divider != 0) {
          s_Data_Output[threadIdx.y][offset + i] = RGB210::pack(accR / divider, accG / divider, accB / divider,
                                                                RGB210::a(s_Data_Input[threadIdx.y][offset + i]));
        } else {
          s_Data_Output[threadIdx.y][offset + i] = 0;
        }
      }
    }
    __syncthreads();
    // write to global memory (coalesced access)
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
      if ((i * ROWS_BLOCKDIM_X + baseX) < width) {
        OutputWithHaloOffset[i * ROWS_BLOCKDIM_X] = s_Data_Output[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////

__constant__ uint32_t c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(uint32_t* h_Kernel) {
  cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(uint32_t));
}

template <bool wrap>
__global__ void convolutionRowsKernel(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, int width,
                                      int height, int pitch) {
  __shared__ uint32_t s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

  // Offset to the left halo edge
  const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  // Load main data
#pragma unroll
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (width - baseX > i * ROWS_BLOCKDIM_X) ? src[i * ROWS_BLOCKDIM_X]
                                              : (wrap ? src[i * ROWS_BLOCKDIM_X - baseX] : src[width - 1 - baseX]);
  }

  // Load left halo
#pragma unroll
  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (baseX >= -i * ROWS_BLOCKDIM_X) ? src[i * ROWS_BLOCKDIM_X]
                                        : (wrap ? src[width - baseX - i * ROWS_BLOCKDIM_X] : src[-baseX]);
  }

  // Load right halo
#pragma unroll
  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (width - baseX > i * ROWS_BLOCKDIM_X) ? src[i * ROWS_BLOCKDIM_X]
                                              : (wrap ? src[i * ROWS_BLOCKDIM_X - baseX] : src[width - 1 - baseX]);
  }

  // Compute and store results
  __syncthreads();

#pragma unroll
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    uint32_t accR = 0;
    uint32_t accG = 0;
    uint32_t accB = 0;
    uint32_t divider = 0;
#pragma unroll
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      uint32_t v = s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
      const int32_t isSolid = !!RGBA::a(v);
      accR += isSolid * c_Kernel[KERNEL_RADIUS - j] * RGBA::r(v);
      accG += isSolid * c_Kernel[KERNEL_RADIUS - j] * RGBA::g(v);
      accB += isSolid * c_Kernel[KERNEL_RADIUS - j] * RGBA::b(v);
      divider += isSolid * c_Kernel[KERNEL_RADIUS - j];
    }

    if (width - baseX > i * COLUMNS_BLOCKDIM_X) {
      dst[i * ROWS_BLOCKDIM_X] = (divider == 0)
                                     ? 0
                                     : RGBA::pack(accR / divider, accG / divider, accB / divider,
                                                  RGBA::a(s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X]));
    }
  }
}

__global__ void convolutionColumnsKernel(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, int width,
                                         int height, int pitch) {
  __shared__ uint32_t
      s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
  const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  // Main data
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (height - baseY > i * COLUMNS_BLOCKDIM_Y)
                                                                    ? src[i * COLUMNS_BLOCKDIM_Y * pitch]
                                                                    : src[(height - 1 - baseY) * pitch];
  }

  // Upper halo
#pragma unroll
  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? src[i * COLUMNS_BLOCKDIM_Y * pitch] : src[-baseY * pitch];
  }

  // Lower halo
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (height - baseY > i * COLUMNS_BLOCKDIM_Y)
                                                                    ? src[i * COLUMNS_BLOCKDIM_Y * pitch]
                                                                    : src[(height - 1 - baseY) * pitch];
  }

  // Compute and store results
  __syncthreads();
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    uint32_t accR = 0;
    uint32_t accG = 0;
    uint32_t accB = 0;
    uint32_t divider = 0;
#pragma unroll
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      uint32_t v = s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
      const int32_t isSolid = !!RGBA::a(v);
      accR += isSolid * c_Kernel[KERNEL_RADIUS - j] * RGBA::r(v);
      accG += isSolid * c_Kernel[KERNEL_RADIUS - j] * RGBA::g(v);
      accB += isSolid * c_Kernel[KERNEL_RADIUS - j] * RGBA::b(v);
      divider += isSolid * c_Kernel[KERNEL_RADIUS - j];
    }

    if (height - baseY > i * COLUMNS_BLOCKDIM_Y) {
      dst[i * COLUMNS_BLOCKDIM_Y * pitch] =
          (divider == 0) ? 0
                         : RGBA::pack(accR / divider, accG / divider, accB / divider,
                                      RGBA::a(s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]));
    }
  }
}
}  // namespace Image
}  // namespace VideoStitch

#endif
