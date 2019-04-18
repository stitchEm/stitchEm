// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <extensionChecker.hpp>

#include "bmpInput.hpp"
#include "bmpr.hpp"

#include <iostream>
#include <cassert>

namespace VideoStitch {
namespace Input {

BmpReader* BmpReader::create(const std::string& fileNameTemplate, const Plugin::VSReaderPlugin::Config& runtime) {
  BmpReader* lReturn = 0;
  const ProbeResult& probeResult = MultiFileReader::probe(fileNameTemplate);
  if (checkProbeResult(probeResult, runtime)) {
    lReturn = new BmpReader(runtime.id, fileNameTemplate, probeResult, runtime.width, runtime.height);
  }
  return lReturn;
}

ProbeResult BmpReader::probe(const std::string& filename) {
  const ProbeResult& probeResult = MultiFileReader::probe(filename);
  if (!probeResult.valid) {
    return ProbeResult({false, false, -1, -1, -1, -1});
  }
  const std::string& firstFile =
      probeResult.filenameIsTemplate ? filenameFromTemplate(filename, (int)probeResult.firstFrame) : filename;
  std::ostringstream error;
  BMPReader bmpReader(firstFile.c_str(), &error);
  if (!error.str().empty()) {
    Logger::get(Logger::Error) << error.str();
  }
  if (!bmpReader.ok()) {
    return ProbeResult({false, false, -1, -1, -1, -1});
  }
  return ProbeResult({true, probeResult.filenameIsTemplate, probeResult.firstFrame, probeResult.lastFrame,
                      bmpReader.getWidth(), bmpReader.getHeight()});
}

BmpReader::BmpReader(int rid, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t width,
                     int64_t height)
    : Reader(rid),
      MultiFileReader(fileNameTemplate, probeResult, width, height, 3 * width * height, PixelFormat::BGR),
      lineBuffer(new unsigned char[(size_t)(3 * width)]) {}

BmpReader::~BmpReader() { delete[] lineBuffer; }

bool BmpReader::handles(VideoStitch::Ptv::Value const* config) {
  //  bool lReturn = false;
  std::string const* filename = hasStringContent(config);
  if (!filename) {
    return false;
  }

  return (hasExtension(*filename, ".bmp") || hasExtension(*filename, ".BMP"));
}

ReadStatus BmpReader::readFrameInternal(unsigned char* data) {
  const std::string& fn = filenameFromTemplate(curFrame);
  std::ostringstream error;
  BMPReader bmpReader(fn.c_str(), &error);  // FIXME: recycle
  if (!error.str().empty()) {
    Logger::get(Logger::Error) << error.str();
  }
  if (!bmpReader.ok()) {
    std::stringstream msg;
    msg << "Image '" << fn << "': failed to setup reader.";
    return ReadStatus({Origin::Input, ErrType::SetupFailure, msg.str()});
  }
  int64_t jWidth = bmpReader.getWidth();
  int64_t jHeight = bmpReader.getHeight();
  if (getWidth() != jWidth || getHeight() != jHeight) {
    std::stringstream msg;
    msg << "Image '" << fn << "' does not have the right size. Expected " << getWidth() << "x" << getHeight()
        << ", got " << jWidth << "x" << jHeight;
    return ReadStatus({Origin::Input, ErrType::SetupFailure, msg.str()});
  }
  unsigned char* curRow = data;
  for (int64_t row = 0; row < getHeight(); ++row) {
    bmpReader.getNextRow(curRow);
    curRow += 3 * getWidth();
  }
  return Status::OK();
}

void BmpReader::resetDisplayName() { getSpec().setDisplayName(fileNameTemplate.c_str()); }

}  // namespace Input
}  // namespace VideoStitch
