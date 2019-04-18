// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PGMOUTPUTWRITER_HPP_
#define PGMOUTPUTWRITER_HPP_

#include "numberedFilesOutput.hpp"
#include <string>

namespace VideoStitch {
namespace Output {
class PgmWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  void writeFrame(const std::string& filename, const char* data);
  PgmWriter(const char* baseName, uint64_t width, uint64_t height, FrameRate framerate, int referenceFrame,
            int numberedNumDigits);
  ~PgmWriter();
};
}  // namespace Output
}  // namespace VideoStitch

#endif
