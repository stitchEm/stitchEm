// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef YUV420POUTPUTWRITER_HPP_
#define YUV420POUTPUTWRITER_HPP_

#include "numberedFilesOutput.hpp"
#include <string>

namespace VideoStitch {
namespace Output {
class Yuv420PWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  void writeFrame(const std::string& filename, const char* data);
  Yuv420PWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int referenceFrame,
                int numberedNumDigits);
  ~Yuv420PWriter();
  PixelFormat getPixelFormat() const;
};
}  // namespace Output
}  // namespace VideoStitch

#endif
