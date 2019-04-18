// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef NUMBEREDFILESOUTPUTWRITER_HPP_
#define NUMBEREDFILESOUTPUTWRITER_HPP_

#include "libvideostitch/stitchOutput.hpp"
#include <string>

namespace VideoStitch {
namespace Output {
/**
 * A writer that writes to numbered files and auto-increments the frame number.
 */
class NumberedFilesWriter : public VideoWriter {
 public:
  /**
   * Multi-file Writers accept an extra parameter. Looks for the field
   * named 'referenceFrame'.
   * @returns 0 (default) if the field is not found or if its type is
   * not INT.
   */
  static int readReferenceFrame(Ptv::Value const& pConfig);

  void pushVideo(const Frame&);

  NumberedFilesWriter(std::string baseName, unsigned width, unsigned height, FrameRate framerate,
                      PixelFormat pixelFormat, int referenceFrame, int numDigits);
  virtual ~NumberedFilesWriter();
  virtual void writeFrame(const std::string& filename, const char* data) = 0;
  virtual const char* getExtension() const = 0;

 private:
  std::string baseName;
  const int referenceFrame;
  const int numDigits;  // number of digits: set 1 for no leading zeros, set 0 to ignore the numbering, any other
                        // positive value to get a zero-prefixed number
};
}  // namespace Output
}  // namespace VideoStitch

#endif
