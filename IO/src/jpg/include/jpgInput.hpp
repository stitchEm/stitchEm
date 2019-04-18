// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "multiFileInput.hpp"
#include <iostream>

namespace VideoStitch {
namespace Input {
/**
 * JPG image reader.
 */
class JpgReader : public MultiFileReader {
 public:
  static JpgReader* create(const std::string& fileNameTemplate, const Plugin::VSReaderPlugin::Config& runtime);
  static ProbeResult probe(const std::string& fileNameTemplate);
  JpgReader(readerid_t rid, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t targetWidth,
            int64_t targetHeight);
  virtual ~JpgReader();

  virtual void getDisplayType(std::ostream& os) const override { os << "JPEG"; }

  static bool handles(VideoStitch::Ptv::Value const* config);

 private:
  virtual ReadStatus readFrameInternal(unsigned char* data) override;
  void resetDisplayName();
  // line buffer for monochrome reading:
  unsigned char* lineBuffer;
};
}  // namespace Input
}  // namespace VideoStitch
