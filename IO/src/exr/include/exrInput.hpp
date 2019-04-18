// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "multiFileInput.hpp"

#include <iostream>

namespace VideoStitch {
namespace Input {
/**
 * EXR image reader.
 */
class ExrReader : public MultiFileReader {
 public:
  static ExrReader* create(Ptv::Value const* config, Plugin::VSReaderPlugin::Config const& runtime);
  static ProbeResult probe(const std::string& fileNameTemplate);
  ExrReader(readerid_t id, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t targetWidth,
            int64_t targetHeight);
  virtual ~ExrReader();

  virtual void getDisplayType(std::ostream& os) const override { os << "EXR"; }

  static bool handles(Ptv::Value const* config);

 private:
  void resetDisplayName();
  static Status readHeader(const char* filename, int64_t& width, int64_t& height);
  virtual ReadStatus readFrameInternal(unsigned char* data) override;
};
}  // namespace Input
}  // namespace VideoStitch
