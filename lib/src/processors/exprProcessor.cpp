// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exprProcessor.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/render/numberDrafter.hpp"
#include "gpu/render/render.hpp"
#include "gpu/stream.hpp"
#include "util/expression.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/parse.hpp"

#include <iostream>
#include <sstream>
#include <algorithm>

#define DEFAULT_SCALE 0.6
#define DEFAULT_COLOR Image::RGBA::pack(0x00, 160, 220, 0xff)
#define DEFAULT_FILL_COLOR Image::RGBA::pack(0x00, 0x00, 0x00, 0x00)

namespace VideoStitch {
namespace Core {
ExprProcedure* ExprProcedure::create(const Ptv::Value& config) {
  double scale = DEFAULT_SCALE;
  uint32_t color = DEFAULT_COLOR;
  uint32_t bgColor = DEFAULT_FILL_COLOR;
  if (Parse::populateDouble("ExprProcedure", config, "scale", scale, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  // parse colors.
  if (Parse::populateColor("ExprProcedure", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ExprProcedure", config, "bg_color", bgColor, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  std::string value;
  if (Parse::populateString("ExprProcedure", config, "value", value, true) != Parse::PopulateResult_Ok) {
    return NULL;
  }
  Util::Expr* expr = Util::Expr::parse(value);
  if (!expr) {
    Logger::get(Logger::Error) << "ExprProcedure: Cannot parse expression '" << value << "'." << std::endl;
    return NULL;
  }
  return new ExprProcedure(expr, scale, color, bgColor);
}

ExprProcedure::ExprProcedure(Util::Expr* expr, double scale, uint32_t color, uint32_t bgColor)
    : scale(scale), color(color), bgColor(bgColor), expr(expr) {}

void ExprProcedure::getDisplayName(std::ostream& os) const { expr->print(os); }

ExprProcedure::~ExprProcedure() { delete expr; }

namespace {
class AContext : public Util::Context {
 public:
  AContext(const int frame, const readerid_t inputId) : frame(frame), inputId(inputId) {}

  Util::EvalResult get(const std::string& var) const {
    if (var == "cFrame") {
      return Util::EvalResult(frame);
    } else if (var == "inputId") {
      return Util::EvalResult(inputId);
    } else {
      return Util::EvalResult();
    }
  }

 private:
  const int frame;
  const readerid_t inputId;
};
}  // namespace

void ExprProcedure::process(frameid_t frame, GPU::Buffer<uint32_t> devBuffer, int64_t width, int64_t height,
                            readerid_t inputId) const {
  // First fill with bg color.
  if (bgColor & 0xff000000) {
    Render::fillBuffer(devBuffer, bgColor, width, height, GPU::Stream::getDefault());
  }

  Util::EvalResult evalResult = expr->eval(AContext(frame, inputId));
  if (!evalResult.isValid()) {
    return;
  }
  const int numberToDraw = (int)evalResult.getInt();
  int numDigits = 1;
  for (int i = 10; i <= numberToDraw; i *= 10) {
    ++numDigits;
  }
#define INTERDIGIT_MULT 1.1f
  // Use as much as scale of the space, vertically or horizontally.
  const float totalWidth =
      std::min((float)(scale * (double)width),
               (1.0f + INTERDIGIT_MULT * (float)(numDigits - 1)) *
                   Render::NumberDrafter::getNumberWidthForHeight((float)(scale * (double)height)));
  Render::NumberDrafter drafter(totalWidth / ((float)numDigits * INTERDIGIT_MULT));
  const float left = ((float)width - totalWidth) / 2.0f;
  const float top = ((float)height - drafter.getNumberHeight()) / 2.0f;
  int curDigit = numDigits - 1;
  if (numberToDraw == 0) {
    drafter.draw0(devBuffer, width, height, left, top, color, GPU::Stream::getDefault());
  } else {
    for (int tmp = numberToDraw; tmp != 0; tmp /= 10, --curDigit) {
      drafter.draw(tmp % 10, devBuffer, width, height, left + ((float)curDigit * totalWidth) / (float)numDigits, top,
                   color, GPU::Stream::getDefault());
    }
  }
  GPU::Stream::getDefault().synchronize();
}
Status ExprProcedure::process(frameid_t frame, GPU::Surface& surf, int64_t width, int64_t height, readerid_t inputId,
                              GPU::Stream& stream) const {
  Util::EvalResult evalResult = expr->eval(AContext(frame, inputId));
  if (!evalResult.isValid()) {
    return {Origin::PreProcessor, ErrType::InvalidConfiguration, "Invalid expression in preprocessor"};
  }
  const int numberToDraw = (int)evalResult.getInt();
  int numDigits = 1;
  for (int i = 10; i <= numberToDraw; i *= 10) {
    ++numDigits;
  }
#define INTERDIGIT_MULT 1.1f
  // Use as much as scale of the space, vertically or horizontally.
  const float totalWidth =
      std::min((float)(scale * (double)width),
               (1.0f + INTERDIGIT_MULT * (float)(numDigits - 1)) *
                   Render::NumberDrafter::getNumberWidthForHeight((float)(scale * (double)height)));
  Render::NumberDrafter drafter(totalWidth / ((float)numDigits * INTERDIGIT_MULT));
  const float left = ((float)width - totalWidth) / 2.0f;
  const float top = ((float)height - drafter.getNumberHeight()) / 2.0f;
  int curDigit = numDigits - 1;
  if (numberToDraw == 0) {
    drafter.draw0(surf, width, height, left, top, color, stream);
  } else {
    for (int tmp = numberToDraw; tmp != 0; tmp /= 10, --curDigit) {
      drafter.draw(tmp % 10, surf, width, height, left + ((float)curDigit * totalWidth) / (float)numDigits, top, color,
                   stream);
    }
  }
  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
