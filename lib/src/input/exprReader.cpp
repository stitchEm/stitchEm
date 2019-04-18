// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exprReader.hpp"

#include "backend/common/imageOps.hpp"
#include "processors/exprProcessor.hpp"
#include "processors/gridProcessor.hpp"
#include "util/expression.hpp"
#include "checkerBoardReader.hpp"
#include "colorReader.hpp"
#include "exprReader.hpp"
#include "movingCheckerReader.hpp"
#include "profilingReader.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#define DEFAULT_SCALE 0.6
#define DEFAULT_COLOR Image::RGBA::pack(0xff, 0x00, 0x00, 0xff)
#define DEFAULT_FILL_COLOR Image::RGBA::pack(0x00, 0x00, 0x00, 0xff)
#define DEFAULT_GRID_SIZE 32
#define DEFAULT_GRID_LINE_WIDTH 2
#define DEFAULT_GRID_COLOR Image::RGBA::pack(0xff, 0x00, 0x00, 0xff)
#define DEFAULT_GRID_BG_COLOR Image::RGBA::pack(0x00, 0x00, 0x00, 0xff)

namespace VideoStitch {
namespace Input {

VideoReader* ProceduralReader::create(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                      int64_t targetHeight) {
  if (!config.has("name")) {
    return NULL;
  }
  const std::string& name = config.has("name")->asString();
  if (name == "frameNumber") {
    return createFrameNumberReader(id, config, targetWidth, targetHeight);
  } else if (name == "expr") {
    return createExpressionReader(id, config, targetWidth, targetHeight);
  } else if (name == "grid") {
    return createGridReader(id, config, targetWidth, targetHeight);
  } else if (name == "checker") {
    return new CheckerBoardReader(id, config, targetWidth, targetHeight);
  } else if (name == "color") {
    return new ColorReader(id, config, targetWidth, targetHeight);
  } else if (name == "profiling") {
    return new ProfilingReader(id, targetWidth, targetHeight);
  } else if (name == "movingChecker") {
    return new MovingCheckerReader(id, targetWidth, targetHeight);
  }

  Logger::get(Logger::Error) << "Error: no such procedural reader: '" << name << "'." << std::endl;
  return NULL;
}

bool ProceduralReader::isKnown(const Ptv::Value& config) {
  if (!config.has("name")) {
    return false;
  }
  const std::string& name = config.has("name")->asString();
  if (name == "frameNumber" || name == "expr" || name == "grid" || name == "checker" || name == "color") {
    return true;
  }
  return false;
}

ProceduralReader* ProceduralReader::createExpressionReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                                           int64_t targetHeight) {
  if (!config.has("value")) {
    Logger::get(Logger::Error) << "Missing 'value' for ProcessorReader." << std::endl;
    return NULL;
  }
  const std::string& exprValue = config.has("value")->asString();
  Util::Expr* expr = Util::Expr::parse(exprValue);
  if (!expr) {
    Logger::get(Logger::Error) << "ProcessorReader: Cannot parse '" << exprValue << "'." << std::endl;
    return NULL;
  }
  double scale = DEFAULT_SCALE;
  uint32_t color = DEFAULT_COLOR;
  uint32_t bgColor = DEFAULT_FILL_COLOR;
  if (Parse::populateDouble("ReaderConfig", config, "scale", scale, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ReaderConfig", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ReaderConfig", config, "bg_color", bgColor, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  return new ProceduralReader(id, new Core::ExprProcedure(expr, scale, color, bgColor), targetWidth, targetHeight);
}

ProceduralReader* ProceduralReader::createFrameNumberReader(readerid_t id, const Ptv::Value& config,
                                                            int64_t targetWidth, int64_t targetHeight) {
  double scale = DEFAULT_SCALE;
  uint32_t color = DEFAULT_COLOR;
  uint32_t bgColor = DEFAULT_FILL_COLOR;
  if (Parse::populateDouble("ReaderConfig", config, "scale", scale, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ReaderConfig", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ReaderConfig", config, "bg_color", bgColor, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  return new ProceduralReader(id, new Core::ExprProcedure(new Util::ContextExpr("cFrame"), scale, color, bgColor),
                              targetWidth, targetHeight);
}

ProceduralReader* ProceduralReader::createGridReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                                     int64_t targetHeight) {
  int size = DEFAULT_GRID_SIZE;
  int lineWidth = DEFAULT_GRID_LINE_WIDTH;
  uint32_t color = DEFAULT_GRID_COLOR;
  uint32_t bgColor = DEFAULT_GRID_BG_COLOR;
  if (Parse::populateInt("ReaderConfig", config, "size", size, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateInt("ReaderConfig", config, "line_width", lineWidth, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ReaderConfig", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  if (Parse::populateColor("ReaderConfig", config, "bg_color", bgColor, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  return new ProceduralReader(id, new Core::GridProcedure(size, lineWidth, color, bgColor), targetWidth, targetHeight);
}

ProceduralReader::ProceduralReader(readerid_t id, Procedure* processor, int64_t targetWidth, int64_t targetHeight)
    : Reader(id),
      VideoReader(targetWidth, targetHeight, targetWidth * targetHeight * sizeof(uint32_t), RGBA, Device,
                  {60, 1} /*fps*/, 0, NO_LAST_FRAME, true /* procedural */, NULL),
      processor(processor),
      curDate(-1) {
  std::stringstream ss;
  processor->getDisplayName(ss);
  getSpec().setDisplayName(ss.str().c_str());
}

ProceduralReader::~ProceduralReader() { delete processor; }

Status ProceduralReader::seekFrame(frameid_t) { return Status::OK(); }

ReadStatus ProceduralReader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  // XXX TODO FIXME procedurals with a frame rate please
  curDate += (mtime_t)round(getSpec().frameRate.den / (double)getSpec().frameRate.num * 1000000.0);
  date = curDate;
  int frameId =
      (int)round((double)curDate * (double)getSpec().frameRate.num / (double)getSpec().frameRate.den / 1000000.0);
  processor->process(frameId, GPU::Buffer<uint32_t>::wrap((uint32_t*)videoFrame, getWidth() * getHeight()), getWidth(),
                     getHeight(), id);
  // Everything is done on the GPU
  return ReadStatus::OK();
}

namespace {
class Context : public Util::Context {
  Util::EvalResult get(const std::string& /*var*/) const { return Util::EvalResult(); }
};
}  // namespace
}  // namespace Input
}  // namespace VideoStitch
