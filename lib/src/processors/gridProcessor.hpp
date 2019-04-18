// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"

#include "input/exprReader.hpp"

#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"

#include <iostream>

namespace VideoStitch {

namespace Core {

/**
 * @brief A processor that writes the result of evaluating an expression.
 *
 * Available variables are:
 *  - inputId: The input id.
 *  - cFrame: The current frame.
 *  - rFrame: The current reader frame.
 */
class GridProcedure : public Input::Procedure {
 public:
  static GridProcedure* create(const Ptv::Value& config);
  GridProcedure(int size, int lineWidth, uint32_t color, uint32_t bgColor);
  ~GridProcedure();
  void process(frameid_t frame, GPU::Buffer<uint32_t> buffer, int64_t width, int64_t height, readerid_t inputId) const;
  void getDisplayName(std::ostream& os) const { os << "Procedural(P): Grid"; }

 private:
  const int size;
  const int lineWidth;
  const uint32_t color;
  const uint32_t bgColor;
};
}  // namespace Core
}  // namespace VideoStitch
