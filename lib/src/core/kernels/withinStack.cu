// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CUDAWITHINSTACK_H_
#define CUDAWITHINSTACK_H_

#ifndef VS_OPENCL
#include <cuda_runtime.h>
#include <math_constants.h>
#endif

#include <cmath>

#ifndef __CUDACC__
#include <algorithm>
#include <math.h>
#endif

namespace VideoStitch {
namespace Core {

/**
 * Input must be within both texture and crop rect
 */
inline __device__ bool isWithinRect(const float2 uv, float width, float height) {
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height;
}

inline __device__ bool isWithinCropRect(const float2 uv, float width, float height, float cLeft, float cRight,
                                        float cTop, float cBottom) {
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height && cLeft <= uv.x && uv.x <= cRight &&
         cTop <= uv.y && uv.y <= cBottom;
}

inline __device__ bool isWithinCropCircle(const float2 uv, float width, float height, float cLeft, float cRight,
                                          float cTop, float cBottom) {
  const float centerX = (cRight + cLeft) / 2.0f;
  const float centerY = (cBottom + cTop) / 2.0f;
  const float radius = fminf(cRight - cLeft, cBottom - cTop) / 2.0f;
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height &&
         (uv.x - centerX) * (uv.x - centerX) + (uv.y - centerY) * (uv.y - centerY) < radius * radius;
}

namespace TransformStack {

inline __host__ bool isWithinCropRect(const float2 uv, float width, float height, float cLeft, float cRight, float cTop,
                                      float cBottom) {
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height && cLeft <= uv.x && uv.x < cRight &&
         cTop <= uv.y && uv.y < cBottom;
}

inline __host__ bool isWithinCropCircle(const float2 uv, float width, float height, float cLeft, float cRight,
                                        float cTop, float cBottom) {
  const float centerX = (cRight + cLeft) / 2.0f;
  const float centerY = (cBottom + cTop) / 2.0f;
  const float radius = fminf(cRight - cLeft, cBottom - cTop) / 2.0f;
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height &&
         (uv.x - centerX) * (uv.x - centerX) + (uv.y - centerY) * (uv.y - centerY) < radius * radius;
}

}  // namespace TransformStack

#ifdef __CUDACC__
/**
 * Output cropper with a rectangular shape.
 */
struct OutputRectCropper {
 public:
  static inline __device__ bool isPanoPointVisible(int x, int y, int panoWidth, int panoHeight) { return true; }
};

/**
 * Output cropper with a circular shape.
 */
struct OutputCircleCropper {
 public:
  static inline __device__ bool isPanoPointVisible(int x, int y, int panoWidth, int panoHeight) {
    // We want to be in a frame where values are at the pixel centers instead of pixels top-left.
    // That would mean using (x + 0.5, y + 0.5) instead of (x, y). Since we want integers, we multiply everything by 2
    // to yield (2 * x + 1, 2 * y + 1).
    x = 2 * x + 1;
    y = 2 * y + 1;

    // The radius is simply the smallest of the semi-axis. For an odd size, we always ignore the last pixel.
    const int centerX = panoWidth & (~1);
    const int centerY = panoHeight & (~1);
    int radiusSquared = min(centerX, centerY);
    radiusSquared *= radiusSquared;
    return (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY) <= radiusSquared;
  }
};
#endif

}  // namespace Core
}  // namespace VideoStitch

#endif
