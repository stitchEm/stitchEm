// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BMPINPUT_HPP_
#define BMPINPUT_HPP_

#include "multiFileInput.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputFactory.hpp"

#include <iostream>
#include <string>

namespace VideoStitch {
namespace Input {
/**
 * BMP image reader.
 */
class BmpReader : public MultiFileReader {
 public:
  static BmpReader* create(const std::string& fileNameTemplate, const Plugin::VSReaderPlugin::Config& runtime);
  static ProbeResult probe(const std::string& fileNameTemplate);
  BmpReader(int rid, const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t targetWidth,
            int64_t targetHeight);
  virtual ~BmpReader();
  void getDisplayType(std::ostream& os) const { os << "BMP"; }

  static bool handles(VideoStitch::Ptv::Value const* config);

 private:
  virtual ReadStatus readFrameInternal(unsigned char* data);
  void resetDisplayName();
  // line buffer for monochrome reading:
  unsigned char* lineBuffer;
};
}  // namespace Input
}  // namespace VideoStitch

#endif
