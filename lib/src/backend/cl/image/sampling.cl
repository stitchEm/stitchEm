// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backend/cl/core1/warpKernelDef.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

// ------------------ Downsampling

#include "backend/common/image/sampling.gpu"

// ------------------ Upsampling

/**
 * Upsample @src by a factor of two on each dimension and put it in into @dst.
 * @dst has size (@dstWidth * @dstHeight), @dst has size ((@dstWidth + 1)/2 * (@dstHeight + 1)/2).
 * This is more complex than subsampling since we need to interpolate at the same time.
 * We use shared memory to share reads to global memory between threads.
 * In addition, we make sure that memory accesses are coalesced.
 * To avoid divergence in the regular case, there are two kernels: one that applies inside
 * the image, and one that applies to boundaries.
 * The alpha is taken to be solid if at least one sample is solid.
 */

/**
// Bilinear interpolation.
//                           +=======+=======+=======+
//                           |       |       |       |
//                           |   A   |   B   |   C   |
//  |       |       |        |       |       |       |
//  +=======+=======+=       +=======+===+===+=======+
//  |       |       |        |       | a | b |       |
//  |   D   |   E   |        |   D   +---+---+   F   |
//  |       |       |        |       | c | d |       |
//  +=======+=======+=  =>   +=======+===+===+=======+
//  |       |       |        |       |       |       |
//  |   G   |   H   |        |   G   |   H   |   I   |
//  |       |       |        |       |       |       |
//  +=======+=======+=       +=======+=======+=======+
//
// The current thread loads source pixel E, then computes interpolated values for a, b, c, d:
//    a = 1 / 16 * A + 3 / 16 * [D + B] + 9 / 16 * E
//    b = 1 / 16 * C + 3 / 16 * [B + F] + 9 / 16 * E
//    c = 1 / 16 * G + 3 / 16 * [D + H] + 9 / 16 * E
//    d = 1 / 16 * I + 3 / 16 * [F + H] + 9 / 16 * E
*/

__device__ uint32_t interpolateRGB210(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  // see above
  const int32_t alphaA = Image_RGB210_a(a);
  const int32_t alphaB = Image_RGB210_a(b);
  const int32_t alphaC = Image_RGB210_a(c);
  const int32_t alphaD = Image_RGB210_a(d);
  const int32_t divisor = 9 * alphaA + 3 * (alphaB + alphaC) + alphaD;
  return Image_RGB210_pack(
      (alphaA * 9 * Image_RGB210_r(a) + 3 * (alphaB * Image_RGB210_r(b) + alphaC * Image_RGB210_r(c)) +
       alphaD * Image_RGB210_r(d)) /
          divisor,
      (alphaA * 9 * Image_RGB210_g(a) + 3 * (alphaB * Image_RGB210_g(b) + alphaC * Image_RGB210_g(c)) +
       alphaD * Image_RGB210_g(d)) /
          divisor,
      (alphaA * 9 * Image_RGB210_b(a) + 3 * (alphaB * Image_RGB210_b(b) + alphaC * Image_RGB210_b(c)) +
       alphaD * Image_RGB210_b(d)) /
          divisor,
      divisor > 0);
}

__device__ uint32_t interpolateRGBA(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  // see above
  const uint32_t alphaA = !!Image_RGBA_a(a);
  const uint32_t alphaB = !!Image_RGBA_a(b);
  const uint32_t alphaC = !!Image_RGBA_a(c);
  const uint32_t alphaD = !!Image_RGBA_a(d);
  const uint32_t divisor = 9 * alphaA + 3 * (alphaB + alphaC) + alphaD;
  if (divisor) {
    return Image_RGBA_pack((alphaA * 9 * Image_RGBA_r(a) + 3 * (alphaB * Image_RGBA_r(b) + alphaC * Image_RGBA_r(c)) +
                            alphaD * Image_RGBA_r(d)) /
                               divisor,
                           (alphaA * 9 * Image_RGBA_g(a) + 3 * (alphaB * Image_RGBA_g(b) + alphaC * Image_RGBA_g(c)) +
                            alphaD * Image_RGBA_g(d)) /
                               divisor,
                           (alphaA * 9 * Image_RGBA_b(a) + 3 * (alphaB * Image_RGBA_b(b) + alphaC * Image_RGBA_b(c)) +
                            alphaD * Image_RGBA_b(d)) /
                               divisor,
                           0xff);
  } else {
    return 0;
  }
}

__device__ unsigned char interpolate(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
  // see above
  return (unsigned char)(9.0f / 16.0f * a + 3.0f / 16.0f * (b + c) + 1.0f / 16.0f * d);
}

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

kernel void upsample22KernelScalar(global_mem unsigned char* __restrict__ dst,
                                   const global_mem unsigned char* __restrict__ src, unsigned dstWidth,
                                   unsigned dstHeight, unsigned srcWidth, unsigned srcHeight, int wrap) {
  const unsigned srcX = (unsigned)get_global_id(0);
  const unsigned srcY = (unsigned)get_global_id(1);
  int idX = (int)get_local_id(0);
  int idY = (int)get_local_id(1);

  __local unsigned char localMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
  localMem[idX + 1][idY + 1] = src[srcWidth * srcY + srcX];
  if (idX == 0) {
    localMem[0][idY] = srcX > 0 ? src[srcWidth * srcY + (srcX - 1)]
                                : (wrap ? src[srcWidth * srcY + (srcWidth - 1)] : src[srcWidth * srcY]);
  }
  if (idX == BLOCK_SIZE - 1) {
    localMem[BLOCK_SIZE + 1][idY] = srcX < srcWidth - 1
                                        ? src[srcWidth * srcY + (srcX + 1)]
                                        : (wrap ? src[srcWidth * srcY] : src[srcWidth * srcY + (srcWidth - 1)]);
  }
  if (idY == 0) {
    localMem[idX][0] = srcY > 0 ? src[srcWidth * (srcY - 1) + srcX] : src[srcX];
  }
  if (idY == BLOCK_SIZE - 1) {
    localMem[idX][BLOCK_SIZE + 1] =
        srcY < srcHeight - 1 ? src[srcWidth * (srcY + 1) + srcX] : src[srcWidth * (srcHeight - 1) + srcX];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (srcX < srcWidth && srcY < srcHeight) {
    const unsigned dstX = 2 * srcX;
    const unsigned dstY = 2 * srcY;

    const unsigned char A = localMem[idX - 1][idY - 1];
    const unsigned char B = localMem[idX][idY - 1];
    const unsigned char C = localMem[idX + 1][idY - 1];
    const unsigned char D = localMem[idX - 1][idY];
    const unsigned char E = localMem[idX][idY];
    const unsigned char F = localMem[idX + 1][idY];
    const unsigned char G = localMem[idX - 1][idY + 1];
    const unsigned char H = localMem[idX][idY + 1];
    const unsigned char I = localMem[idX + 1][idY + 1];
    if (dstX < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX] = interpolate(E, D, B, A);
    }
    if (dstX + 1 < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX + 1] = interpolate(E, F, B, C);
    }
    if (dstX < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX] = interpolate(E, D, H, G);
    }
    if (dstX + 1 < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX + 1] = interpolate(E, F, H, I);
    }
  }
}

kernel void upsample22KernelRGBA(global_mem uint32_t* __restrict__ dst, const global_mem uint32_t* __restrict__ src,
                                 unsigned dstWidth, unsigned dstHeight, unsigned srcWidth, unsigned srcHeight,
                                 int wrap) {
  const unsigned srcX = (unsigned)get_global_id(0);
  const unsigned srcY = (unsigned)get_global_id(1);
  int idX = (int)get_local_id(0);
  int idY = (int)get_local_id(1);

  __local uint32_t localMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  // interior
  localMem[idX + 1][idY + 1] = src[srcWidth * srcY + srcX];

  if (idX == 0) {
    // left column
    unsigned fetchX;
    if (srcX > 0) {
      fetchX = srcX - 1;
    } else if (wrap) {
      fetchX = srcWidth - 1;
    } else {
      fetchX = 0;
    }
    localMem[0][idY + 1] = src[srcWidth * srcY + fetchX];

    if (idY == 0) {
      // top-left corner
      unsigned fetchY;
      if (srcY > 0) {
        fetchY = srcY - 1;
      } else {
        fetchY = 0;
      }
      localMem[0][0] = src[srcWidth * fetchY + fetchX];
    } else if (idY == BLOCK_SIZE - 1) {
      // bottom-left corner
      unsigned fetchY;
      if (srcY < srcHeight - 1) {
        fetchY = srcY + 1;
      } else {
        fetchY = srcHeight - 1;
      }
      localMem[0][BLOCK_SIZE + 1] = src[srcWidth * fetchY + fetchX];
    }
  }

  if (idX == BLOCK_SIZE - 1) {
    // right column
    unsigned fetchX;
    if (srcX < srcWidth - 1) {
      fetchX = srcX + 1;
    } else if (wrap) {
      fetchX = 0;
    } else {
      fetchX = srcWidth - 1;
    }
    localMem[BLOCK_SIZE + 1][idY + 1] = src[srcWidth * srcY + fetchX];

    if (idY == 0) {
      // top-right corner
      unsigned fetchY;
      if (srcY > 0) {
        fetchY = srcY - 1;
      } else {
        fetchY = 0;
      }
      localMem[BLOCK_SIZE + 1][0] = src[srcWidth * fetchY + fetchX];
    } else if (idY == BLOCK_SIZE - 1) {
      // bottom-right corner
      unsigned fetchY;
      if (srcY < srcHeight - 1) {
        fetchY = srcY + 1;
      } else {
        fetchY = srcHeight - 1;
      }
      localMem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = src[srcWidth * fetchY + fetchX];
    }
  }

  if (idY == 0) {
    // top row
    localMem[idX + 1][0] = srcY > 0 ? src[srcWidth * (srcY - 1) + srcX] : src[srcX];
  }

  if (idY == BLOCK_SIZE - 1) {
    // bottom row
    localMem[idX + 1][BLOCK_SIZE + 1] =
        srcY < srcHeight - 1 ? src[srcWidth * (srcY + 1) + srcX] : src[srcWidth * (srcHeight - 1) + srcX];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (srcX < srcWidth && srcY < srcHeight) {
    const unsigned dstX = 2 * srcX;
    const unsigned dstY = 2 * srcY;

    const uint32_t A = localMem[idX][idY];
    const uint32_t B = localMem[idX + 1][idY];
    const uint32_t C = localMem[idX + 2][idY];
    const uint32_t D = localMem[idX][idY + 1];
    const uint32_t E = localMem[idX + 1][idY + 1];
    const uint32_t F = localMem[idX + 2][idY + 1];
    const uint32_t G = localMem[idX][idY + 2];
    const uint32_t H = localMem[idX + 1][idY + 2];
    const uint32_t I = localMem[idX + 2][idY + 2];
    if (dstX < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX] = interpolateRGBA(E, D, B, A);
    }
    if (dstX + 1 < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX + 1] = interpolateRGBA(E, F, B, C);
    }
    if (dstX < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX] = interpolateRGBA(E, D, H, G);
    }
    if (dstX + 1 < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX + 1] = interpolateRGBA(E, F, H, I);
    }
  }
}

kernel void upsample22KernelRGB210(global_mem uint32_t* __restrict__ dst, const global_mem uint32_t* __restrict__ src,
                                   unsigned dstWidth, unsigned dstHeight, unsigned srcWidth, unsigned srcHeight,
                                   int wrap) {
  const unsigned srcX = (unsigned)get_global_id(0);
  const unsigned srcY = (unsigned)get_global_id(1);
  int idX = (int)get_local_id(0);
  int idY = (int)get_local_id(1);

  __local uint32_t localMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  // interior
  localMem[idX + 1][idY + 1] = src[srcWidth * srcY + srcX];

  if (idX == 0) {
    // left column
    unsigned fetchX;
    if (srcX > 0) {
      fetchX = srcX - 1;
    } else if (wrap) {
      fetchX = srcWidth - 1;
    } else {
      fetchX = 0;
    }
    localMem[0][idY + 1] = src[srcWidth * srcY + fetchX];

    if (idY == 0) {
      // top-left corner
      unsigned fetchY;
      if (srcY > 0) {
        fetchY = srcY - 1;
      } else {
        fetchY = 0;
      }
      localMem[0][0] = src[srcWidth * fetchY + fetchX];
    } else if (idY == BLOCK_SIZE - 1) {
      // bottom-left corner
      unsigned fetchY;
      if (srcY < srcHeight - 1) {
        fetchY = srcY + 1;
      } else {
        fetchY = srcHeight - 1;
      }
      localMem[0][BLOCK_SIZE + 1] = src[srcWidth * fetchY + fetchX];
    }
  }

  if (idX == BLOCK_SIZE - 1) {
    // right column
    unsigned fetchX;
    if (srcX < srcWidth - 1) {
      fetchX = srcX + 1;
    } else if (wrap) {
      fetchX = 0;
    } else {
      fetchX = srcWidth - 1;
    }
    localMem[BLOCK_SIZE + 1][idY + 1] = src[srcWidth * srcY + fetchX];

    if (idY == 0) {
      // top-right corner
      unsigned fetchY;
      if (srcY > 0) {
        fetchY = srcY - 1;
      } else {
        fetchY = 0;
      }
      localMem[BLOCK_SIZE + 1][0] = src[srcWidth * fetchY + fetchX];
    } else if (idY == BLOCK_SIZE - 1) {
      // bottom-right corner
      unsigned fetchY;
      if (srcY < srcHeight - 1) {
        fetchY = srcY + 1;
      } else {
        fetchY = srcHeight - 1;
      }
      localMem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = src[srcWidth * fetchY + fetchX];
    }
  }

  if (idY == 0) {
    // top row
    localMem[idX + 1][0] = srcY > 0 ? src[srcWidth * (srcY - 1) + srcX] : src[srcX];
  }

  if (idY == BLOCK_SIZE - 1) {
    // bottom row
    localMem[idX + 1][BLOCK_SIZE + 1] =
        srcY < srcHeight - 1 ? src[srcWidth * (srcY + 1) + srcX] : src[srcWidth * (srcHeight - 1) + srcX];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (srcX < srcWidth && srcY < srcHeight) {
    const unsigned dstX = 2 * srcX;
    const unsigned dstY = 2 * srcY;

    const uint32_t A = localMem[idX][idY];
    const uint32_t B = localMem[idX + 1][idY];
    const uint32_t C = localMem[idX + 2][idY];
    const uint32_t D = localMem[idX][idY + 1];
    const uint32_t E = localMem[idX + 1][idY + 1];
    const uint32_t F = localMem[idX + 2][idY + 1];
    const uint32_t G = localMem[idX][idY + 2];
    const uint32_t H = localMem[idX + 1][idY + 2];
    const uint32_t I = localMem[idX + 2][idY + 2];
    if (dstX < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX] = interpolateRGB210(E, D, B, A);
    }
    if (dstX + 1 < dstWidth && dstY < dstHeight) {
      dst[dstWidth * dstY + dstX + 1] = interpolateRGB210(E, F, B, C);
    }
    if (dstX < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX] = interpolateRGB210(E, D, H, G);
    }
    if (dstX + 1 < dstWidth && dstY + 1 < dstHeight) {
      dst[dstWidth * (dstY + 1) + dstX + 1] = interpolateRGB210(E, F, H, I);
    }
  }
}

#pragma clang diagnostic pop
