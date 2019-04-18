// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef TIFFOUTPUTWRITER_HPP_
#define TIFFOUTPUTWRITER_HPP_

#include "numberedFilesOutput.hpp"
#include <string>

#define TIFF_WRITER_DEFAULT_COMPRESSION "none"
#define TIFF_WRITER_DEFAULT_ALPHA "keep"

namespace VideoStitch {
namespace Output {
/**
 * @brief A writer that writes tiff images.
 */
class TiffWriter : public NumberedFilesWriter {
 public:
  static const char extension[];
  const char* getExtension() const { return extension; }
  /**
   * Creates a TiffWriter. Returns false on error.
   */
  static Potential<TiffWriter> create(const Ptv::Value& config, const char* baseName, unsigned width, unsigned height,
                                      FrameRate framerate, int referenceFrame);
  void writeFrame(const std::string& filename, const char* data);
  TiffWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int compression,
             int referenceFrame, int numberedNumDigits);
  ~TiffWriter();
  PixelFormat getPixelFormat() const;

 private:
  int compression;
};
}  // namespace Output
}  // namespace VideoStitch

#endif
