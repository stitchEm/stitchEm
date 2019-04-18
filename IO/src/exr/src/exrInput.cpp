// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exrInput.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "extensionChecker.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif

#ifdef __linux__
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#else
#include <openexr/ImfInputFile.h>
#include <openexr/ImfChannelList.h>
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cassert>
#include <sstream>

namespace VideoStitch {
namespace Input {

ExrReader* ExrReader::create(VideoStitch::Ptv::Value const* config, Plugin::VSReaderPlugin::Config const& runtime) {
  std::string const* fileNameTemplate = hasStringContent(config);
  if (fileNameTemplate) {
    const ProbeResult& probeResult = MultiFileReader::probe(*fileNameTemplate);
    if (checkProbeResult(probeResult, runtime)) {
      return new ExrReader(runtime.id, *fileNameTemplate, probeResult, runtime.width, runtime.height);
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

ProbeResult ExrReader::probe(const std::string& filename) {
  const ProbeResult& probeResult = MultiFileReader::probe(filename);
  if (!probeResult.valid) {
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }
  int64_t width = -1;
  int64_t height = -1;
  std::string firstFile = filename;
  if (probeResult.filenameIsTemplate) {
    firstFile = filenameFromTemplate(filename, int(probeResult.firstFrame));
  }
  if (!readHeader(firstFile.c_str(), width, height).ok()) {
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }
  return ProbeResult({true, probeResult.filenameIsTemplate, probeResult.firstFrame, probeResult.lastFrame, width,
                      height, false, probeResult.hasVideo});
}

ExrReader::ExrReader(readerid_t id, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t width,
                     int64_t height)
    : Reader(id),
      MultiFileReader(fileNameTemplate, probeResult, width, height, 4 * width * height, VideoStitch::RGBA) {}

ExrReader::~ExrReader() {}

bool ExrReader::handles(Ptv::Value const* config) {
  std::string const* filename = hasStringContent(config);
  if (!filename) {
    return false;
  }
  return (hasExtension(*filename, ".exr"));
}

void ExrReader::resetDisplayName() { getSpec().setDisplayName(fileNameTemplate.c_str()); }

Status ExrReader::readHeader(const char* /* filename */, int64_t& /* width */, int64_t& /* height */) {
#if 0
  try {
    Imf::InputFile file(filename);

    // data size
    Imath::Box2i dw = file.header().dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;

    // check channels, RGB must be present, or raises an exception
    const Imf::ChannelList &channels = file.header().channels();
    channels["R"];
    channels["G"];
    channels["B"];
    channels["A"];
  } catch (const std::exception& e) {
    std::stringstream msg;
    msg << "ExrWriter::writeFrame(): " << e.what() << std::endl;
    Logger::get(Logger::Error) << msg.str();
    return { Origin::Input, ErrType::RuntimeError, "[ExrReader] Invalid EXR header" };
  }

  return Status();
#else
  return {Origin::Input, ErrType::UnsupportedAction, "[ExrReader] EXR library not supported for inputs yest"};
#endif
}

ReadStatus ExrReader::readFrameInternal(unsigned char* /* data */) {
  // const std::string& fn = filenameFromTemplate(curFrame);
  // Casting to void* is OK here since data has been allocated by cuda and is therefore correctly aligned.

  return Status(Origin::Input, ErrType::UnsupportedAction, "[ExrReader] EXR library not supported for inputs yest");
}

}  // namespace Input
}  // namespace VideoStitch
