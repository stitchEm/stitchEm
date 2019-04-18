// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "yuv420pOutput.hpp"

#include "pnm.hpp"
#include <sstream>
#include <iostream>

#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Output {
const char Yuv420PWriter::extension[] = "yuv";

void Yuv420PWriter::writeFrame(const std::string& filename, const char* data) {
  {
    std::stringstream ss;
    ss << filename << "-y.pgm";
    std::ofstream* ofs =
        Util::PpmWriter::openPgm(ss.str().c_str(), getWidth(), getHeight(), &Logger::get(Logger::Error));
    if (!ofs) {
      return;
    }
    ofs->write(data, getWidth() * getHeight());
    data += getWidth() * getHeight();
    delete ofs;
  }
  int ssWidth = (getWidth() + 1) / 2;
  int ssHeight = (getHeight() + 1) / 2;
  {
    std::stringstream ss;
    ss << filename << "-u.pgm";
    std::ofstream* ofs = Util::PpmWriter::openPgm(ss.str().c_str(), ssWidth, ssHeight, &Logger::get(Logger::Error));
    if (!ofs) {
      return;
    }
    ofs->write(data, ssWidth * ssHeight);
    delete ofs;
    data += ssWidth * ssHeight;
  }
  {
    std::stringstream ss;
    ss << filename << "-v.pgm";
    std::ofstream* ofs = Util::PpmWriter::openPgm(ss.str().c_str(), ssWidth, ssHeight, &Logger::get(Logger::Error));
    if (!ofs) {
      return;
    }
    ofs->write(data, ssWidth * ssHeight);
    delete ofs;
  }
}

Yuv420PWriter::Yuv420PWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate,
                             int referenceFrame, int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, width, height, framerate, PixelFormat::YV12, referenceFrame, numberedNumDigits) {}

Yuv420PWriter::~Yuv420PWriter() {}
}  // namespace Output
}  // namespace VideoStitch
