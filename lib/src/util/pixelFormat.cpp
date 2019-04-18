// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/frame.hpp"

namespace VideoStitch {

const char* getStringFromPixelFormat(const PixelFormat pixelFormat) {
  switch (pixelFormat) {
    case RGBA:
      return "RGBA";
    case RGB:
      return "RGB";
    case BGR:
      return "BGR";
    case BGRU:
      return "BGRU";
    case UYVY:
      return "UYVY";
    case YUY2:
      return "YUY2";
    case YV12:
      return "YV12";
    case Grayscale:
      return "Gray scale";
    case Grayscale16:
      return "Gray scale (16 Bit)";
    case F32_C1:
      return "Depth";
    case DEPTH:
      return "Depth YV12";
    default:
      return "";
  }
}

PixelFormat getPixelFormatFromString(const std::string& name) {
  if (name == "RGBA") {
    return PixelFormat::RGBA;
  } else if (name == "RGB") {
    return PixelFormat::RGB;
  } else if (name == "BGR") {
    return PixelFormat::BGR;
  } else if (name == "BGRU") {
    return PixelFormat::BGRU;
  } else if (name == "UYVY") {
    return PixelFormat::UYVY;
  } else if (name == "YUY2") {
    return PixelFormat::YUY2;
  } else if (name == "YV12") {
    return PixelFormat::YV12;
  } else if (name == "Gray scale") {
    return PixelFormat::Grayscale;
  } else if (name == "Gray scale (16 Bit)") {
    return PixelFormat::Grayscale16;
  } else if (name == "Depth") {
    return PixelFormat::F32_C1;
  } else if (name == "Depth YV12") {
    return PixelFormat::DEPTH;
  } else {
    return PixelFormat::Unknown;
  }
}

int32_t getFrameDataSize(int32_t width, int32_t height, const PixelFormat pixelFormat) {
  switch (pixelFormat) {
    // 32 bpp
    case RGBA:
    case BGRU:
    case YUV422P10:  // 20 bpp but 10 bits values are padded to 16 bits
    case F32_C1:
      return width * height * 4;
    // 24 bpp
    case RGB:
    case BGR:
      return width * height * 3;
    // 16 bpp
    case UYVY:
    case YUY2:
    case Grayscale16:
      return width * height * 2;
    // 12 bpp
    case YV12:
    case NV12:
    case DEPTH:
      return (width * height * 3) / 2;
    // 8 bpp
    case Grayscale:
      return (width * height);
    default:
      return 0;
  }
}

}  // namespace VideoStitch
