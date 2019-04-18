// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "detail/Png.hpp"

#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace detail {

Status Png::readHeader(const char* filename, int64_t& width, int64_t& height) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, filename)) {
    return {Origin::Input, ErrType::RuntimeError,
            "[PNGReader] Could not open '" + std::string(filename) + "': " + image.message};
  }
  width = image.width;
  height = image.height;
  return Status::OK();
#else
  return {Origin::Input, ErrType::UnsupportedAction, "[PNGReader] PNG library not supported"};
#endif
}

Status Png::readRGBAFromFile(const char* filename, int64_t width, int64_t height, void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, filename)) {
    return {Origin::Input, ErrType::RuntimeError,
            "[PNGReader] Could not open '" + std::string(filename) + "': " + image.message};
  }
  image.format = PNG_FORMAT_RGBA;
  if ((int64_t)image.width != width || (int64_t)image.height != height) {
    std::stringstream msg;
    msg << "[PNGReader] Unexpected size. Got " << image.width << " x " << image.height << ", expected " << width
        << " x " << height;
    return {Origin::Input, ErrType::RuntimeError, msg.str()};
  }
  if (!png_image_finish_read(&image, NULL, data, 0, NULL)) {
    return {Origin::Input, ErrType::RuntimeError,
            "[PNGReader] Failed reading '" + std::string(filename) + "': " + image.message};
  }
  return Status::OK();
#else
  return {Origin::Input, ErrType::UnsupportedAction, "[PNGReader] PNG library not supported"};
#endif
}

Status Png::writeRGBToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_RGB;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(3 * width), NULL)) {
    return {Origin::Input, ErrType::RuntimeError,
            "[PNGReader] Could not write to '" + std::string(filename) + "': " + image.message};
  }
  return Status::OK();
#else
  return {Origin::Input, ErrType::UnsupportedAction, "[PNGReader] PNG library not supported"};
#endif
}

Status Png::writeRGBAToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_RGBA;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(4 * width), NULL)) {
    return {Origin::Input, ErrType::RuntimeError,
            "[PNGReader] Could not write to '" + std::string(filename) + "': " + image.message};
  }
  return Status::OK();
#else
  return {Origin::Input, ErrType::UnsupportedAction, "[PNGReader] PNG library not supported"};
#endif
}

Status Png::writeMonochrome16ToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_GRAY | PNG_FORMAT_FLAG_LINEAR;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(width), NULL)) {
    return {Origin::Input, ErrType::RuntimeError,
            "[PNGReader] Could not write to '" + std::string(filename) + "': " + image.message};
  }
  return Status::OK();
#else
  return {Origin::Input, ErrType::UnsupportedAction, "[PNGReader] PNG library not supported"};
#endif
}

}  // namespace detail
}  // namespace VideoStitch
