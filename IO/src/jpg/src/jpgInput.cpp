// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"

#include "jpgInput.hpp"
#include "jpg.hpp"

#include "extensionChecker.hpp"
#include <sstream>
#include <cassert>

namespace VideoStitch {
namespace Input {

JpgReader* JpgReader::create(const std::string& fileNameTemplate, const Plugin::VSReaderPlugin::Config& runtime) {
  const ProbeResult& probeResult = MultiFileReader::probe(fileNameTemplate);
  if (checkProbeResult(probeResult, runtime)) {
    return new JpgReader(runtime.id, fileNameTemplate, probeResult, runtime.width, runtime.height);
  } else {
    return nullptr;
  }
}

ProbeResult JpgReader::probe(const std::string& filename) {
  const ProbeResult& probeResult = MultiFileReader::probe(filename);
  if (!probeResult.valid) {
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }
  std::string firstFile = filename;
  if (probeResult.filenameIsTemplate) {
    firstFile = filenameFromTemplate(filename, int(probeResult.firstFrame));
  }
  const JPGReader jpgReader(firstFile.c_str(), &Logger::get(Logger::Error));
  if (!jpgReader.ok()) {
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }
  return ProbeResult({true, probeResult.filenameIsTemplate, probeResult.firstFrame, probeResult.lastFrame,
                      jpgReader.getWidth(), jpgReader.getHeight(), false, probeResult.hasVideo});
}

JpgReader::JpgReader(readerid_t rid, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t width,
                     int64_t height)
    : Reader(rid),
      MultiFileReader(fileNameTemplate, probeResult, width, height, 3 * width * height, VideoStitch::RGB),
      lineBuffer(new unsigned char[(size_t)(3 * width)]) {}

JpgReader::~JpgReader() { delete[] lineBuffer; }

bool JpgReader::handles(VideoStitch::Ptv::Value const* config) {
  std::string const* filename = hasStringContent(config);
  if (!filename) {
    return false;
  }
  return (hasExtension(*filename, ".jpg") || hasExtension(*filename, ".jpeg"));
}

ReadStatus JpgReader::readFrameInternal(unsigned char* data) {
  const std::string& fn = filenameFromTemplate(curFrame);
  // TODOLATERSTATUS: pipe jpeg log into Status message
  JPGReader jpgReader(fn.c_str(), &Logger::get(Logger::Error));  // FIXME: recycle
  if (!jpgReader.ok()) {
    std::stringstream msg;
    msg << "Image '" << fn << "': failed to setup reader.";
    return ReadStatus({Origin::Input, ErrType::SetupFailure, msg.str()});
  }
  int64_t jWidth = jpgReader.getWidth();
  int64_t jHeight = jpgReader.getHeight();
  if (getWidth() != jWidth || getHeight() != jHeight) {
    std::stringstream msg;
    msg << "Image '" << fn << "' does not have the right size. Expected " << getWidth() << "x" << getHeight()
        << ", got " << jWidth << "x" << jHeight;
    return ReadStatus({Origin::Input, ErrType::SetupFailure, msg.str()});
  }
  unsigned char* curRow = data;
  for (int64_t row = 0; row < getHeight(); ++row) {
    jpgReader.getNextRow(curRow);
    curRow += 3 * getWidth();
  }
  return ReadStatus::OK();
}

void JpgReader::resetDisplayName() { getSpec().setDisplayName(fileNameTemplate.c_str()); }
}  // namespace Input
}  // namespace VideoStitch
