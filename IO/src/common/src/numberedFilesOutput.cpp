// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/ptv.hpp"
#include "numberedFilesOutput.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>

namespace VideoStitch {
namespace Output {
void NumberedFilesWriter::pushVideo(const Frame& data) {
  std::stringstream ss;
  ss << baseName;
  FrameRate fr = getFrameRate();
  frameid_t frId = (frameid_t)round(((double)data.pts / 1000000) * (fr.num / (double)fr.den));
  if (numDigits > 0) {
    ss << "-" << std::setfill('0') << std::setw(numDigits) << (referenceFrame + frId);
  }
  ss << "." << getExtension();
  this->writeFrame(ss.str(), (const char*)data.planes[0]);
}

NumberedFilesWriter::NumberedFilesWriter(std::string baseName, unsigned width, unsigned height, FrameRate framerate,
                                         PixelFormat pixelFormat, int referenceFrame, int numDigits)
    : Output(""),
      VideoWriter(width, height, framerate, pixelFormat),
      baseName(baseName),
      referenceFrame(referenceFrame),
      numDigits(numDigits) {}

NumberedFilesWriter::~NumberedFilesWriter() {}

int NumberedFilesWriter::readReferenceFrame(Ptv::Value const& pConfig) {
  int64_t lRefFr = 0;
  Ptv::Value const* field = pConfig.has("referenceFrame");
  if (field) {
    if (field->getType() == Ptv::Value::INT) {
      lRefFr = field->asInt();
    }
  }
  return int(lRefFr);
}
}  // namespace Output
}  // namespace VideoStitch
