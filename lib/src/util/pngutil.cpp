// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pngutil.hpp"

#include "libvideostitch/logging.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>

namespace VideoStitch {
namespace Util {
#ifndef PNGLIB_UNSUPPORTED
png_image PngReader::image;
#endif

PngReader::PngReader() {}

bool PngReader::readRGBAFromFile(const char* filename, int64_t width, int64_t height, void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, filename)) {
    Logger::get(Logger::Error) << "PngReader: Could not open " << filename << ": " << image.message << std::endl;
    return false;
  }
  image.format = PNG_FORMAT_RGBA;
  if ((int64_t)image.width != width || (int64_t)image.height != height) {
    Logger::get(Logger::Error) << "PngReader: Unexpected size. Got " << image.width << " x " << image.height
                               << std::endl;
    return false;
  }
  if (!png_image_finish_read(&image, NULL, data, 0, NULL)) {
    Logger::get(Logger::Error) << "PngReader: " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool PngReader::readRGBAFromFile(const char* filename, int64_t& width, int64_t& height,
                                 std::vector<unsigned char>& data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, filename)) {
    Logger::get(Logger::Error) << "PngReader: Could not open " << filename << ": " << image.message << std::endl;
    return false;
  }
  image.format = PNG_FORMAT_RGBA;
  width = (int64_t)image.width;
  height = (int64_t)image.height;
  data.resize(width * height * 4);
  if (!png_image_finish_read(&image, nullptr, &data[0], 0, nullptr)) {
    Logger::get(Logger::Error) << "PngReader: " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool PngReader::readMonochromeFromFile(const char* filename, int64_t width, int64_t height, void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, filename)) {
    Logger::get(Logger::Error) << "PngReader: Could not open " << filename << ": " << image.message << std::endl;
    return false;
  }
  image.format = PNG_FORMAT_GRAY;
  if ((int64_t)image.width != width || (int64_t)image.height != height) {
    Logger::get(Logger::Error) << "PngReader: Unexpected size. Got " << image.width << " x " << image.height
                               << std::endl;
    return false;
  }
  if (!png_image_finish_read(&image, NULL, data, 0, NULL)) {
    Logger::get(Logger::Error) << "PngReader: " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool PngReader::readMaskFromMemory(const unsigned char* input, size_t inputLen, int64_t& width, int64_t& height,
                                   void** data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&image, reinterpret_cast<const png_const_voidp*>(input),
                                        (png_size_t)inputLen)) {
    Logger::get(Logger::Error) << "PngReader: Could not read from memory: " << image.message << std::endl;
    return false;
  }
  if (image.format != (PNG_FORMAT_FLAG_COLOR | PNG_FORMAT_FLAG_COLORMAP)) {
    Logger::get(Logger::Error) << "PngReader: Unexpected color format '" << image.format << "'." << std::endl;
    return false;
  }
  if (image.colormap_entries > 3) {
    Logger::get(Logger::Error) << "PngReader: Expected <= 3 colormap entries, got '" << image.colormap_entries << "'."
                               << std::endl;
    return false;
  }
  *data = malloc(image.width * image.height);
  width = image.width;
  height = image.height;
  unsigned char* colormap = new unsigned char[PNG_IMAGE_COLORMAP_SIZE(image)];
  png_color bgColor;
  bgColor.red = 0;
  bgColor.green = 0;
  bgColor.blue = 0;
  if (!png_image_finish_read(&image, &bgColor, *data, 0, colormap)) {
    Logger::get(Logger::Error) << "PngReader: " << image.message << std::endl;
    delete[] colormap;
    return false;
  }
  // Make sure that black is 0, red is 1. We do not support other colors, make them black by default.
  unsigned char lookupTable[] = {0, 0, 0};
  int black_map = -1;
  int red_map = -1;
  int green_map = -1;
  int other_map = -1;
  for (int i = 0; i < (int)image.colormap_entries; ++i) {
    switch ((int)colormap[3 * i] + 256 * (int)colormap[3 * i + 1] + 256 * 256 * (int)colormap[3 * i + 2]) {
      case 0:  // black
        black_map = i;
        break;
      case 255:  // red
        red_map = i;
        lookupTable[i] = 1;
        break;
      case 255 * 256:  // green
        green_map = i;
        break;
      default:
        Logger::get(Logger::Warning) << "Found unexpected color in mask: (" << colormap[3 * i] << ","
                                     << colormap[3 * i + 1] << "," << colormap[3 * i + 2] << "), ignoring..."
                                     << std::endl;
        other_map = 0;
        break;
    }
  }
  delete[] colormap;
  // We found no red, this is useless.
  if (red_map == -1) {
    Logger::get(Logger::Warning) << "Found a mask but no masked parts." << std::endl;
    return false;
  }
  // std::cout << (int)lookupTable[0] << " " << (int)lookupTable[1] << " " << (int)lookupTable[2] << std::endl;
  // If we are lucky, we may already have the colors in order, with only black and red. Else we need to do a lookup
  // pass.
  if (!(black_map == 0 && red_map == 1 && green_map == -1 && other_map == -1)) {
    unsigned char* indices = static_cast<unsigned char*>(*data);
    for (unsigned i = 0; i < width * height; ++i) {
      indices[i] = lookupTable[indices[i]];
    }
  }
  return true;
#else
  return false;
#endif
}

#ifndef PNGLIB_UNSUPPORTED
namespace {
/**
 * Custom png write function.
 */
void pngWriteData(png_structp png_ptr, png_bytep data, png_uint_32 length) {
  std::string* output = (std::string*)png_get_io_ptr(png_ptr);
  output->append((const char*)data, (size_t)length);
}

/**
 * Custom png flush function.
 */
void pngFlushData(png_structp) {}
}  // namespace
#endif

#if defined GCC_VERSION && !defined __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclobbered"
#endif
bool PngReader::writeMaskToMemory(std::string& output, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  output.clear();
  // Check that we have only supported color.
  for (int64_t i = 0; i < width * height; ++i) {
    if (((const unsigned char*)data)[i] > 1) {
      Logger::get(Logger::Warning) << "Unsupported mask color " << ((const unsigned char*)data)[i] << std::endl;
      return false;
    }
  }
  png_struct* png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) {
    return false;
  }
  png_info* pngInfo = png_create_info_struct(png);
  if (!pngInfo) {
    png_destroy_write_struct(&png, NULL);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &pngInfo);
    return false;
  }
  // Set our custom write function.
  png_set_write_fn(png, (png_voidp)&output, (png_rw_ptr)pngWriteData, (png_flush_ptr)pngFlushData);
  // png_set_write_user_transform_fn();
  png_set_benign_errors(png, 0 /*error*/);

  png_set_IHDR(png, pngInfo, (png_uint_32)width, (png_uint_32)height,
               1,  // bpc, two colors == 1 bit
               PNG_COLOR_TYPE_PALETTE, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_COMPRESSION_TYPE_DEFAULT);
  // Set the colormap ("palette").
  png_color colormap[2];
  colormap[0].red = 0;
  colormap[0].green = 0;
  colormap[0].blue = 0;
  colormap[1].red = 255;
  colormap[1].green = 0;
  colormap[1].blue = 0;
  png_set_PLTE(png, pngInfo, colormap, 2);
  png_write_info(png, pngInfo);
  // Tell libppng to byte-pack our 1 bits.
  png_set_packing(png);

  png_byte* row = (png_byte*)data;
  for (int64_t i = 0; i < height; ++i) {
    png_write_row(png, row);
    row += width;
  }
  png_write_end(png, pngInfo);
  png_destroy_write_struct(&png, &pngInfo);
  return true;
#else
  return false;
#endif
}

#if defined __GNUC__ && !defined __clang__
#pragma GCC diagnostic pop
#endif

bool PngReader::writeRGBAToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_RGBA;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(4 * width), NULL)) {
    Logger::get(Logger::Error) << "PngWriter: Could not write to " << filename << ": " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool PngReader::writeBGRToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_BGR;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(3 * width), NULL)) {
    Logger::get(Logger::Error) << "PngWriter: Could not write to " << filename << ": " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool PngReader::writeMonochromToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_GRAY;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(width), NULL)) {
    Logger::get(Logger::Error) << "PngWriter: Could not write to " << filename << ": " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool PngReader::writeMonochrome16ToFile(const char* filename, int64_t width, int64_t height, const void* data) {
#ifndef PNGLIB_UNSUPPORTED
  std::memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = PNG_FORMAT_GRAY | PNG_FORMAT_FLAG_LINEAR;
  if (!png_image_write_to_file(&image, filename, 0, data, (png_uint_32)(width), NULL)) {
    Logger::get(Logger::Error) << "PngWriter: Could not write to " << filename << ": " << image.message << std::endl;
    return false;
  }
  return true;
#else
  return false;
#endif
}

}  // namespace Util
}  // namespace VideoStitch
