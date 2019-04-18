// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pgmOutput.hpp"
#include "pnm.hpp"
#include "libvideostitch/logging.hpp"

#include <sstream>
#include <iostream>

namespace VideoStitch {
namespace Output {
const char PgmWriter::extension[] = "pgm";

void PgmWriter::writeFrame(const std::string& filename, const char* data) {
  std::ofstream* ofs = Util::PpmWriter::openPgm(filename.c_str(), getWidth(), getHeight(), &Logger::get(Logger::Error));
  ofs->write(data, getWidth() * getHeight());
  delete ofs;
}

PgmWriter::PgmWriter(const char* baseName, uint64_t width, uint64_t height, FrameRate framerate, int referenceFrame,
                     int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, (unsigned)width, (unsigned)height, framerate, PixelFormat::Grayscale,
                          referenceFrame, numberedNumDigits) {}

PgmWriter::~PgmWriter() {}
}  // namespace Output
}  // namespace VideoStitch
