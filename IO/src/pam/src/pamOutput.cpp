// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/logging.hpp"
#include "pnm.hpp"
#include "pamOutput.hpp"
#include "pgmOutput.hpp"
#include "ppmOutput.hpp"
#include "yuv420pOutput.hpp"

#include <sstream>
#include <iostream>

namespace VideoStitch {
namespace Output {
const char PamWriter::extension[] = "pam";

Output* PamWriter::create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config run_time) {
  VideoWriter* l_return = 0;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    int referenceFrame = VideoStitch::Output::NumberedFilesWriter::readReferenceFrame(*config);
    if (!strcmp(baseConfig.strFmt, "pam")) {
      l_return = new PamWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate, referenceFrame,
                               baseConfig.numberNumDigits);
    } else if (!strcmp(baseConfig.strFmt, "pgm")) {
      l_return = new PgmWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate, referenceFrame,
                               baseConfig.numberNumDigits);
    } else if (!strcmp(baseConfig.strFmt, "ppm")) {
      l_return = new PpmWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate, referenceFrame,
                               baseConfig.numberNumDigits);
    } else if (!strcmp(baseConfig.strFmt, "yuv420p")) {
      l_return = new Yuv420PWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate,
                                   referenceFrame, baseConfig.numberNumDigits);
    }
  }
  return l_return;
}

bool PamWriter::handles(Ptv::Value const* config) {
  bool lReturn = false;
  VideoStitch::Output::BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    lReturn = ((!strcmp(baseConfig.strFmt, "pam")) || (!strcmp(baseConfig.strFmt, "pgm")) ||
               (!strcmp(baseConfig.strFmt, "ppm")) || (!strcmp(baseConfig.strFmt, "yuv420p")));
  }
  return lReturn;
}

void PamWriter::writeFrame(const std::string& filename, const char* data) {
  std::ofstream* ofs = Util::PpmWriter::openPam(filename.c_str(), getWidth(), getHeight(), &Logger::get(Logger::Error));
  ofs->write(data, getWidth() * getHeight() * 4);
  delete ofs;
}

PamWriter::PamWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int referenceFrame,
                     int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, width, height, framerate, PixelFormat::RGBA, referenceFrame, numberedNumDigits) {}

PamWriter::~PamWriter() {}
}  // namespace Output
}  // namespace VideoStitch
