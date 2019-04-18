// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ppmOutput.hpp"
#include "pnm.hpp"
#include "libvideostitch/logging.hpp"

#include <sstream>
#include <iostream>

namespace VideoStitch {
namespace Output {
const char PpmWriter::extension[] = "ppm";

void PpmWriter::writeFrame(const std::string& filename, const char* data) {
  std::ofstream* ofs = Util::PpmWriter::openPpm(filename.c_str(), getWidth(), getHeight(), &Logger::get(Logger::Error));
  ofs->write(data, getWidth() * getHeight() * 3);
  delete ofs;
}

PpmWriter::PpmWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int referenceFrame,
                     int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, width, height, framerate, PixelFormat::RGB, referenceFrame, numberedNumDigits) {}

PpmWriter::~PpmWriter() {}
}  // namespace Output
}  // namespace VideoStitch
