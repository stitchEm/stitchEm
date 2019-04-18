// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "numberedFilesOutput.hpp"
#include <string>

namespace VideoStitch {
namespace Output {
class PpmWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  void writeFrame(const std::string& filename, const char* data);
  PpmWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int referenceFrame,
            int numberedNumDigits);
  ~PpmWriter();
};
}  // namespace Output
}  // namespace VideoStitch
