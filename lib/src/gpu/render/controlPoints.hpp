// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"
#include "gpu/vectorTypes.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Render {

/**
 * @brief A class to draw control points in CUDA buffers.
 */
class ControlPointRenderer {
 public:
  /**
   * Create a Control Point Renderer (shape of a cross).
   * @param thickness The thickness of each line, in fractional pixels.
   * @param length The length of each line
   */
  ControlPointRenderer(float size, float thickness);

  /**
   * Draw a control point point.
   * @param p point to draw
   * @param dst Destination buffer.
   * @param dstWidth Buffer thickness.
   * @param dstHeight Buffer height.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  void drawControlPoint(const int2& p, GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, uint32_t color,
                        GPU::Stream stream) const;

 private:
  const float size;
  const float thickness;
};
}  // namespace Render
}  // namespace VideoStitch
