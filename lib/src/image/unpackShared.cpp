// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "unpack.hpp"

#include <gpu/memcpy.hpp>

#include <cassert>

namespace VideoStitch {
namespace Image {

Status unpackCommonPixelFormat(PixelFormat format, GPU::Surface& dst, GPU::Buffer<const unsigned char> src,
                               std::size_t width, std::size_t height, GPU::Stream stream) {
  switch (format) {
    case VideoStitch::RGBA:
      return GPU::memcpyAsync(dst, src.as<const uint32_t>(), stream);
    case VideoStitch::RGB:
      return Image::convertRGBToRGBA(dst, src, width, height, stream);
      /*
    case VideoStitch::BGR:
      return Image::convertBGRToRGBA(dst, src, width, height, stream);
    case VideoStitch::BGRU:
      return Image::convertBGRUToRGBA(dst, src, width, height, stream);
      */
    case VideoStitch::YUV422P10:
      return Image::convertYUV422P10ToRGBA(dst, src, width, height, stream);
    case VideoStitch::UYVY:
      return Image::convertUYVYToRGBA(dst, src, width, height, stream);
    case VideoStitch::YUY2:
      return Image::convertYUY2ToRGBA(dst, src, width, height, stream);
    case VideoStitch::YV12:
      return Image::convertYV12ToRGBA(dst, src, width, height, stream);
    case VideoStitch::NV12:
      return Image::convertNV12ToRGBA(dst, src, width, height, stream);
    case VideoStitch::Grayscale:
      return Image::convertGrayscaleToRGBA(dst, src, width, height, stream);
      /*
    case Bayer_RGGB:
      return Image::convertBayerRGGBToRGBA(dst, src, width, height, stream);
    case Bayer_BGGR:
      return Image::convertBayerBGGRToRGBA(dst, src, width, height, stream);
    case Bayer_GBRG:
      return Image::convertBayerGBRGToRGBA(dst, src, width, height, stream);
    case Bayer_GRBG:
      return Image::convertBayerGRBGToRGBA(dst, src, width, height, stream);
      */
    case VideoStitch::Unknown:
    default:
      assert(false);
      return {Origin::Stitcher, ErrType::ImplementationError, "Cannot unpack unknown pixel format"};
  }
}

}  // namespace Image
}  // namespace VideoStitch
