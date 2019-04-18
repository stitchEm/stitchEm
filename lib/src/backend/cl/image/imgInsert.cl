// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "imageFormat.h"

#include "../gpuKernelDef.h"

// TODO_OPENCL_IMPL: shared with CUDA implementation

/**
 * This kernel inserts the content of the (packed) image @src at offset (@offsetX, offsetY) into @dest.
 * On overflow, the image wraps if hWrap (resp vWrap) is true. Else nothing is written.
 * Pixels with zero alpha are not merged.
 * 2D version: We assume that the @src (but not the dst) image is divisible
 * by the block size on each dimension.
 */
#define DEFINE_IMGIIK(funcName, testPredicate, dstIndexComputation)                                            \
  __global__ void funcName(global_mem uint32_t* __restrict__ dst, unsigned dstWidth, unsigned dstHeight,       \
                           global_mem const uint32_t* __restrict__ src, unsigned srcWidth, unsigned srcHeight, \
                           unsigned offsetX, unsigned offsetY) {                                               \
    const unsigned int srcX = (unsigned int)get_global_id(0);                                                  \
    const unsigned int srcY = (unsigned int)get_global_id(1);                                                  \
                                                                                                               \
    unsigned dstX = srcX + offsetX;                                                                            \
    unsigned dstY = srcY + offsetY;                                                                            \
                                                                                                               \
    if (srcX < srcWidth && srcY < srcHeight) {                                                                 \
      uint32_t p = src[srcWidth * srcY + srcX];                                                                \
      if (testPredicate) {                                                                                     \
        dst[dstIndexComputation] = p;                                                                          \
      }                                                                                                        \
    }                                                                                                          \
  }

/**
 * Same as above, except that @mask is used for blending.
 */
#define DEFINE_IMGIIKM(funcName, testPredicate, dstIndexComputation, r, g, b, a, pack)                          \
  void kernel funcName(global unsigned int* dst, unsigned int dstWidth, unsigned int dstHeight,                 \
                       const global unsigned int* src, unsigned srcWidth, unsigned srcHeight, unsigned offsetX, \
                       unsigned offsetY, const global unsigned char* mask) {                                    \
    const unsigned int srcX = (unsigned int)get_global_id(0);                                                   \
    const unsigned int srcY = (unsigned int)get_global_id(1);                                                   \
                                                                                                                \
    const unsigned int dstX = srcX + offsetX;                                                                   \
    const unsigned int dstY = srcY + offsetY;                                                                   \
                                                                                                                \
    if (srcX < srcWidth && srcY < srcHeight) {                                                                  \
      const unsigned int srcIndex = srcWidth * srcY + srcX;                                                     \
      const unsigned int p = src[srcIndex];                                                                     \
      if (testPredicate) {                                                                                      \
        const unsigned int dstIndex = dstIndexComputation;                                                      \
        unsigned int q = dst[dstIndex];                                                                         \
        if (a(q)) {                                                                                             \
          const int m = mask[srcIndex];                                                                         \
          const unsigned int mR = (m * r(p) + (255 - m) * r(q)) / 255;                                          \
          const unsigned int mG = (m * g(p) + (255 - m) * g(q)) / 255;                                          \
          const unsigned int mB = (m * b(p) + (255 - m) * b(q)) / 255;                                          \
          q = pack(mR, mG, mB, 0xff);                                                                           \
        } else {                                                                                                \
          if (mask[srcIndex]) {                                                                                 \
            q = p;                                                                                              \
          }                                                                                                     \
        }                                                                                                       \
        dst[dstIndex] = q;                                                                                      \
      }                                                                                                         \
    }                                                                                                           \
  }

// /**
//  * The blending mask is overlaid on the image with the given color.
//  * The value is blended with @dst such that if mask is either 0 or 255, we keep the original value,
//  * and if the mask is 128, we use the full opaque color.
//  *
//  *  (128 - abs(m - 128))
//  */
// #define DEFINE_IMGISIK(funcName, testPredicate, dstIndexComputation) \
// __global__ void \
// funcName(uint32_t* __restrict__ dst, unsigned dstWidth, unsigned dstHeight, unsigned srcWidth, unsigned srcHeight, \
//          unsigned offsetX, unsigned offsetY, const unsigned char* __restrict__ mask) \
// { \
//   unsigned srcX = blockIdx.x*blockDim.x + threadIdx.x; \
//   unsigned srcY = blockIdx.y*blockDim.y + threadIdx.y; \
//  \
//   unsigned dstX = srcX + offsetX; \
//   unsigned dstY = srcY + offsetY; \
//  \
//   if (srcX < srcWidth && srcY < srcHeight) { \
//     unsigned srcIndex = srcWidth * srcY + srcX; \
//     if (testPredicate) { \
//       unsigned dstIndex = dstIndexComputation; \
//       uint32_t q = dst[dstIndex]; \
//       if (RGB210::a(q)) { \
//         int32_t m = abs((int)mask[srcIndex] - 128); \
//         uint32_t mR = (m * RGB210::r(q) + (128 - m) * 0xff) / 128; \
//         uint32_t mG = (m * RGB210::g(q) + (128 - m) * 0x00) / 128; \
//         uint32_t mB = (m * RGB210::b(q) + (128 - m) * 0x00) / 128; \
//         q = RGB210::pack(mR, mG, mB, 0xff); \
//       } \
//       dst[dstIndex] = q; \
//     } \
//   } \
// }

DEFINE_IMGIIKM(imgInsertIntoKernelMaskedNoWrap, Image_RGBA_a(p) && dstX < dstWidth && dstY < dstHeight,
               dstWidth* dstY + dstX, Image_RGBA_r, Image_RGBA_g, Image_RGBA_b, Image_RGBA_a, Image_RGBA_pack)
DEFINE_IMGIIK(imgInsertIntoKernelNoWrap, Image_RGBA_a(p) && dstX < dstWidth && dstY < dstHeight, dstWidth* dstY + dstX)
DEFINE_IMGIIKM(imgInsertIntoKernelMaskedNoWrap10bit, Image_RGB210_a(p) && dstX < dstWidth && dstY < dstHeight,
               dstWidth* dstY + dstX, Image_RGB210_r, Image_RGB210_g, Image_RGB210_b, Image_RGB210_a, Image_RGB210_pack)
DEFINE_IMGIIK(imgInsertIntoKernelNoWrap10bit, Image_RGB210_a(p) && dstX < dstWidth && dstY < dstHeight,
              dstWidth* dstY + dstX)
DEFINE_IMGIIKM(imgInsertIntoKernelMaskedHWrap, Image_RGBA_a(p) && dstY < dstHeight, dstWidth* dstY + (dstX % dstWidth),
               Image_RGBA_r, Image_RGBA_g, Image_RGBA_b, Image_RGBA_a, Image_RGBA_pack)
DEFINE_IMGIIK(imgInsertIntoKernelHWrap, Image_RGBA_a(p) && dstY < dstHeight, dstWidth* dstY + (dstX % dstWidth))
DEFINE_IMGIIKM(imgInsertIntoKernelMaskedHWrap10bit, Image_RGB210_a(p) && dstY < dstHeight,
               dstWidth* dstY + (dstX % dstWidth), Image_RGB210_r, Image_RGB210_g, Image_RGB210_b, Image_RGB210_a,
               Image_RGB210_pack)
DEFINE_IMGIIK(imgInsertIntoKernelHWrap10bit, Image_RGB210_a(p) && dstY < dstHeight, dstWidth* dstY + (dstX % dstWidth))
