// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "numberedFilesOutput.hpp"
#include "libvideostitch/plugin.hpp"
#include <string>

extern "C" {
#include "jpeglib.h"
}

#define JPEG_WRITER_DEFAULT_QUALITY 90

namespace VideoStitch {
namespace Output {
class JpgWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  static Potential<JpgWriter> create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config run_time);
  static bool handles(VideoStitch::Ptv::Value const* config);

  void writeFrame(const std::string& filename, const char* data);
  JpgWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int jpgQuality,
            int referenceFrame, int numberedNumDigits);
  ~JpgWriter();

 private:
  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
};

}  // namespace Output
}  // namespace VideoStitch
