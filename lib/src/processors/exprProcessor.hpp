// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"

#include "input/exprReader.hpp"

#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/ptv.hpp"

#include <string>

namespace VideoStitch {

class ThreadSafeOstream;

namespace Util {
class Expr;
}
namespace Core {

/**
 * @brief A processor that writes the result of evaluating an expression.
 *
 * Available variables are:
 *  - inputId: The input id.
 *  - cFrame: The current frame.
 *  - rFrame: The current reader frame.
 */
class ExprProcedure : public Input::Procedure, public PreProcessor {
 public:
  static ExprProcedure* create(const Ptv::Value& config);
  ExprProcedure(Util::Expr* expr, double scale, uint32_t color, uint32_t bgColor);
  ~ExprProcedure();
  void process(frameid_t frame, GPU::Buffer<uint32_t> buffer, int64_t width, int64_t height, readerid_t inputId) const;
  Status process(frameid_t frame, GPU::Surface& surface, int64_t width, int64_t height, readerid_t inputId,
                 GPU::Stream& stream) const;
  void getDisplayName(std::ostream& os) const;

 private:
  const double scale;
  const uint32_t color;
  const uint32_t bgColor;
  Util::Expr* const expr;
  std::string name;
};
}  // namespace Core
}  // namespace VideoStitch
