// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <gpu/render/controlPoints.hpp>
#include <gpu/render/geometry.hpp>

namespace VideoStitch {
namespace Render {

ControlPointRenderer::ControlPointRenderer(float size, float thickness) : size(size), thickness(thickness) {}

void ControlPointRenderer::drawControlPoint(const int2& p, GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight,
                                            uint32_t color, GPU::Stream stream) const {
  // Horizontal line of the cross
  const float axH = ((float)p.x - size / 2.0f) >= 0 ? ((float)p.x - size / 2.0f) : 0.f;
  const float ayH = (float)p.y;
  const float bxH = ((float)p.x + size / 2.0f) <= (float)dstWidth ? ((float)p.x + size / 2.0f) : (float)dstWidth;
  const float byH = (float)p.y;
  // Vertical line of the cross
  const float axV = (float)p.x;
  const float ayV = ((float)p.y - size / 2.0f) >= 0 ? ((float)p.y - size / 2.0f) : 0.f;
  const float bxV = (float)p.x;
  const float byV = ((float)p.y + size / 2.0f) <= (float)dstHeight ? ((float)p.y + size / 2.0f) : (float)dstHeight;
  drawLine(dst, dstWidth, dstHeight, axH, ayH, bxH, byH, thickness, color, stream);
  drawLine(dst, dstWidth, dstHeight, axV, ayV, bxV, byV, thickness, color, stream);
}

}  // namespace Render
}  // namespace VideoStitch
