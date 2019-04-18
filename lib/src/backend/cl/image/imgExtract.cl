// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

/**
 * This kernel extract a part of the content of the image \p src at
 * offset (\p offsetX, \p offsetY) with size (\p dstWidth x \p dstHeight) and writes it in into a packed buffer \p dst.
 * \p dst must be large enough to hold the \p dstWidth * \p dstHeight pixels.
 * On overflow, the source image wraps if \p hWrap is true. Else pixels are filled with 0.
 * 2D version: We assume that the \p dst (but not the \p src) image is divisible
 * by the block size on each dimension.
 */
void kernel imgExtractFromKernel(global unsigned int* dst, int dstWidth, int dstHeight, const global unsigned int* src,
                                 int srcWidth, int srcHeight, int offsetX, int offsetY, int hWrap) {
  const int dstX = (int)get_global_id(0);
  const int dstY = (int)get_global_id(1);

  const int srcX = offsetX + dstX;
  const int srcY = offsetY + dstY;

  unsigned int res = 0;
  if (dstX < dstWidth && dstY < dstHeight) {
    if (0 <= srcY && srcY < srcHeight) {
      if (hWrap) {
        if (0 <= srcX) {
          if (srcX < srcWidth) {
            res = src[srcWidth * srcY + srcX];
          } else {
            res = src[srcWidth * srcY + (srcX % srcWidth)];
          }
        } else {
          res = src[srcWidth * srcY + srcWidth + (srcX % srcWidth)];  // modulo has sign of dividend
        }
      } else if (0 <= srcX & srcX < srcWidth) {
        res = src[srcWidth * srcY + srcX];
      }
    }
    dst[dstWidth * dstY + dstX] = res;
  }
}
