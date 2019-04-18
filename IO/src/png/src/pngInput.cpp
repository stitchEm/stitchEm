// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pngInput.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "extensionChecker.hpp"
#include <cassert>

namespace VideoStitch {
namespace Input {

PngReader* PngReader::create(VideoStitch::Ptv::Value const* config, Plugin::VSReaderPlugin::Config const& runtime) {
  std::string const* fileNameTemplate = hasStringContent(config);
  if (fileNameTemplate) {
    const ProbeResult& probeResult = MultiFileReader::probe(*fileNameTemplate);
    if (checkProbeResult(probeResult, runtime)) {
      return new PngReader(runtime.id, *fileNameTemplate, probeResult, runtime.width, runtime.height);
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

ProbeResult PngReader::probe(const std::string& filename) {
  const ProbeResult& probeResult = MultiFileReader::probe(filename);
  if (!probeResult.valid) {
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }
  detail::Png png;
  int64_t width = -1;
  int64_t height = -1;
  std::string firstFile = filename;
  if (probeResult.filenameIsTemplate) {
    firstFile = filenameFromTemplate(filename, int(probeResult.firstFrame));
  }
  if (!png.readHeader(firstFile.c_str(), width, height).ok()) {
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }
  return ProbeResult({true, probeResult.filenameIsTemplate, probeResult.firstFrame, probeResult.lastFrame, width,
                      height, false, probeResult.hasVideo});
}

PngReader::PngReader(readerid_t id, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t width,
                     int64_t height)
    : Reader(id),
      MultiFileReader(fileNameTemplate, probeResult, width, height, 4 * width * height, VideoStitch::RGBA) {}

PngReader::~PngReader() {}

bool PngReader::handles(Ptv::Value const* config) {
  std::string const* filename = hasStringContent(config);
  if (!filename) {
    return false;
  }
  return (hasExtension(*filename, ".png"));
}

void PngReader::resetDisplayName() { getSpec().setDisplayName(fileNameTemplate.c_str()); }

ReadStatus PngReader::readFrameInternal(unsigned char* data) {
  const std::string& fn = filenameFromTemplate(curFrame);
  // Casting to void* is OK here since data has been allocated by cuda and is therefore correctly aligned.
  return png.readRGBAFromFile(fn.c_str(), getWidth(), getHeight(), (void*)data);
}

}  // namespace Input
}  // namespace VideoStitch
