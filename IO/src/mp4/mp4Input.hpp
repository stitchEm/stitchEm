// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef MP4INPUT_HPP_
#define MP4INPUT_HPP_

#include "libvideostitch/input.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/inputFactory.hpp"

#include <iostream>
#include <string>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

enum {
  COLOR_FormatYUV420Planar = 0x13,
  COLOR_FormatYUV420SemiPlanar = 0x15,
  COLOR_FormatYCbYCr = 0x19,
  COLOR_FormatAndroidOpaque = 0x7F000789,
  COLOR_QCOM_FormatYUV420SemiPlanar = 0x7fa30c00,
  COLOR_QCOM_FormatYUV420SemiPlanar32m = 0x7fa30c04,
  COLOR_QCOM_FormatYUV420PackedSemiPlanar64x32Tile2m8ka = 0x7fa30c03,
  COLOR_TI_FormatYUV420PackedSemiPlanar = 0x7f000100,
  COLOR_TI_FormatYUV420PackedSemiPlanarInterlaced = 0x7f000001,
};

namespace VideoStitch {
namespace Input {
/**
 * MP4 clip reader.
 */
class Mp4Reader : public VideoReader {
 public:
  static Mp4Reader* create(readerid_t id, const std::string& fileName, const Plugin::VSReaderPlugin::Config& runtime);
  static ProbeResult probe(const std::string& fileName);

  Mp4Reader(readerid_t id, AMediaExtractor* extractor, AMediaCodec* videoCodec, FrameRate frameRate,
            PixelFormat pixelFormat, int32_t stride, int32_t width, int32_t height);
  virtual ~Mp4Reader();

  virtual ReadStatus readFrame(mtime_t&, unsigned char* videoFrame);
  virtual ReadStatus readSamples(size_t nbSamples, Audio::Samples& samples);
  virtual Status seekFrame(frameid_t targetFrame);

  void getDisplayType(std::ostream& os) const { os << "MP4"; }

  static bool handles(const std::string& filename);

 private:
  AMediaExtractor* mExtractor;
  AMediaCodec* mCodec;
  int32_t mWidth;
  int32_t mHeight;
  PixelFormat mPixelFormat;
  int32_t mStride;
  FrameRate mFrameRate;
  bool mStarted;
  struct timeval mTimeval;
};
}  // namespace Input
}  // namespace VideoStitch

#endif
