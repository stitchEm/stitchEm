// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rawOutput.hpp"

#include <cstdint>
#include <sstream>
#include <fstream>
#include <iostream>

#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Output {
const char RawWriter::extension[] = "abgr";

RawWriter* RawWriter::create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config run_time) {
  RawWriter* l_return = nullptr;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    l_return =
        new VideoStitch::Output::RawWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate,
                                           readReferenceFrame(*config), baseConfig.numberNumDigits);
  }
  return l_return;
}

bool RawWriter::handles(Ptv::Value const* config) {
  bool l_return = false;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    l_return = (!strcmp(baseConfig.strFmt, "raw"));
  }
  return l_return;
}

void RawWriter::writeFrame(const std::string& filename, const char* data) {
  std::ofstream ofs(filename, std::ifstream::out | std::ios_base::binary);
  if (!ofs.good()) {
    Logger::get(Logger::Error) << "Cannot open file '" << filename << "' for writing." << std::endl;
  } else {
    {
      uint32_t s = getWidth();
      ofs.write((const char*)&s, 4);
    }
    {
      uint32_t s = getHeight();
      ofs.write((const char*)&s, 4);
    }
    ofs.write(data, getWidth() * getHeight() * 4);
    ofs.close();
  }
}

RawWriter::RawWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int referenceFrame,
                     int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, width, height, framerate, PixelFormat::RGBA, referenceFrame, numberedNumDigits) {}

RawWriter::~RawWriter() {}
}  // namespace Output
}  // namespace VideoStitch
