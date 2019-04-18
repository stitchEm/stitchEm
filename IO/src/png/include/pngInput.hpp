// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "multiFileInput.hpp"
#include "detail/Png.hpp"

#include <iostream>

namespace VideoStitch {
namespace Input {
/**
 * PNG image reader.
 */
class PngReader : public MultiFileReader {
 public:
  static PngReader* create(Ptv::Value const* config, Plugin::VSReaderPlugin::Config const& runtime);
  static ProbeResult probe(const std::string& fileNameTemplate);
  PngReader(readerid_t id, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t targetWidth,
            int64_t targetHeight);
  virtual ~PngReader();

  virtual void getDisplayType(std::ostream& os) const override { os << "PNG"; }

  static bool handles(Ptv::Value const* config);

 private:
  void resetDisplayName();
  virtual ReadStatus readFrameInternal(unsigned char* data) override;
  detail::Png png;
};
}  // namespace Input
}  // namespace VideoStitch
