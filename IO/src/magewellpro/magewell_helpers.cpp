// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "magewell_helpers.hpp"

#include "MWFOURCC.h"

namespace VideoStitch {
namespace Magewell {

DWORD xiColorFormat(const PixelFormat& pixelFormat) {
  switch (pixelFormat) {
    case PixelFormat::BGRU:
      return MWFOURCC_BGRA;
    case PixelFormat::BGR:
      return MWFOURCC_BGR24;
    case PixelFormat::UYVY:
      return MWFOURCC_UYVY;
    case PixelFormat::YUY2:
      return MWFOURCC_YUYV;
    default:
      return MWFOURCC_UNK;
  }
}

}  // namespace Magewell
}  // namespace VideoStitch
