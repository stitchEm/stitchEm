// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "numberedFilesOutput.hpp"
#include "libvideostitch/plugin.hpp"
#include <string>

namespace VideoStitch {
namespace Output {
class PngWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  void writeFrame(const std::string& filename, const char* data);
  PngWriter(const char* baseName, uint64_t width, uint64_t height, FrameRate framerate, PixelFormat pixelFormat,
            int referenceFrame, int numberedNumDigits);
  ~PngWriter();

  /** \name Plug-in API methods */
  //\{
  static PngWriter* create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config const& run_time);
  static bool handles(VideoStitch::Ptv::Value const* config);
  //\}
};
}  // namespace Output
}  // namespace VideoStitch
