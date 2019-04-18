// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gridProcessor.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/processors/grid.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Core {

GridProcedure* GridProcedure::create(const Ptv::Value& config) {
  int size = 32;
  int lineWidth = 2;
  uint32_t color = 0xff0000ff;  // ABGR
  uint32_t bgColor = 0xff000000;
  if (Parse::populateInt("ExprPreProcessor", config, "size", size, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateInt("ExprPreProcessor", config, "lineWidth", lineWidth, false) ==
      Parse::PopulateResult_WrongType) {
    return NULL;
  }
  // parse colors.
  if (Parse::populateColor("ExprPreProcessor", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ExprPreProcessor", config, "bg_color", bgColor, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  return new GridProcedure(size, lineWidth, color, bgColor);
}

GridProcedure::GridProcedure(int size, int lineWidth, uint32_t color, uint32_t bgColor)
    : size(size), lineWidth(lineWidth), color(color), bgColor(bgColor) {}

GridProcedure::~GridProcedure() {}

void GridProcedure::process(frameid_t /*frame*/, GPU::Buffer<uint32_t> buffer, int64_t width, int64_t height,
                            readerid_t /*inputId*/) const {
  if (Image::RGBA::a(bgColor)) {
    if (Image::RGBA::a(color)) {
      grid(buffer, (unsigned)width, (unsigned)height, size, lineWidth, color, bgColor, GPU::Stream::getDefault());
    } else {
      transparentForegroundGrid(buffer, (unsigned)width, (unsigned)height, size, lineWidth, bgColor,
                                GPU::Stream::getDefault());
    }
  } else {
    // Transparent background.
    if (Image::RGBA::a(color)) {
      transparentBackgroundGrid(buffer, (unsigned)width, (unsigned)height, size, lineWidth, color,
                                GPU::Stream::getDefault());
    }
    // Do nothing if both are transparent.
  }
  GPU::Stream::getDefault().synchronize();
}
}  // namespace Core
}  // namespace VideoStitch
