// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exrOutput.hpp"

#include "libvideostitch/logging.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif

#ifdef __linux__
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#else
#include <openexr/ImfOutputFile.h>
#include <openexr/ImfChannelList.h>
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <sstream>
#include <iostream>

#define WRITE_DEPTH_AS_RGB 1

namespace VideoStitch {
namespace Output {
const char ExrWriter::extension[] = "exr";

ExrWriter* ExrWriter::create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config const& run_time) {
  ExrWriter* l_return = nullptr;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    int referenceFrame = readReferenceFrame(*config);
    l_return = new ExrWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate, referenceFrame,
                             baseConfig.numberNumDigits);
  }
  return l_return;
}

bool ExrWriter::handles(VideoStitch::Ptv::Value const* config) {
  bool l_return = false;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    l_return = (!strcmp(baseConfig.strFmt, "exr"));
  }
  return l_return;
}

void ExrWriter::writeFrame(const std::string& filename, const char* data) {
  const float* ptr = (float*)data;
  try {
    Imf::Header header((int)getWidth(), (int)getHeight());
#if WRITE_DEPTH_AS_RGB
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
#else
    header.channels().insert("Z", Imf::Channel(Imf::FLOAT));
#endif

    Imf::OutputFile file(filename.c_str(), header);

    Imf::FrameBuffer frameBuffer;

#if WRITE_DEPTH_AS_RGB
    frameBuffer.insert("R",                                     // name
                       Imf::Slice(Imf::FLOAT,                   // type
                                  (char*)ptr,                   // base
                                  sizeof(*ptr) * 1,             // xStride
                                  sizeof(*ptr) * getWidth()));  // yStride
    frameBuffer.insert("G",                                     // name
                       Imf::Slice(Imf::FLOAT,                   // type
                                  (char*)ptr,                   // base
                                  sizeof(*ptr) * 1,             // xStride
                                  sizeof(*ptr) * getWidth()));  // yStride
    frameBuffer.insert("B",                                     // name
                       Imf::Slice(Imf::FLOAT,                   // type
                                  (char*)ptr,                   // base
                                  sizeof(*ptr) * 1,             // xStride
                                  sizeof(*ptr) * getWidth()));  // yStride
#else
    frameBuffer.insert("Z",                                     // name
                       Imf::Slice(Imf::FLOAT,                   // type
                                  (char*)ptr,                   // base
                                  sizeof(*ptr) * 1,             // xStride
                                  sizeof(*ptr) * getWidth()));  // yStride
#endif

    file.setFrameBuffer(frameBuffer);
    file.writePixels(getHeight());
  } catch (const std::exception& e) {
    std::stringstream msg;
    msg << "ExrWriter::writeFrame(): " << e.what() << std::endl;
    Logger::get(Logger::Error) << msg.str();
  }
}

ExrWriter::ExrWriter(const char* baseName, uint64_t width, uint64_t height, FrameRate framerate, int referenceFrame,
                     int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, (unsigned)width, (unsigned)height, framerate, PixelFormat::F32_C1, referenceFrame,
                          numberedNumDigits) {}

ExrWriter::~ExrWriter() {}
}  // namespace Output
}  // namespace VideoStitch
