// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "numberedFilesOutput.hpp"
#include "libvideostitch/plugin.hpp"
#include <string>

namespace VideoStitch {
namespace Output {
class RawWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  void writeFrame(const std::string& filename, const char* data);
  RawWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int referenceFrame,
            int numberedNumDigits);
  ~RawWriter();
  PixelFormat getPixelFormat() const;

  /** \name Plug-in API methods. */
  //\{
  static RawWriter* create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config run_time);
  static bool handles(Ptv::Value const* config);
  //\}
};
}  // namespace Output
}  // namespace VideoStitch
