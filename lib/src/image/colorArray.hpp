// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// This file defines the host and device utilities for representing pixel arrays.
//

#ifndef COLOR_ARRAY_HPP_
#define COLOR_ARRAY_HPP_

#include "gpu/buffer.hpp"
#include "gpu/vectorTypes.hpp"

#include <stdint.h>
#include <vector>

namespace VideoStitch {
namespace Image {

template <typename T>
inline std::string typeName(void) {
  return "unknown";
}

// See below for details.
struct MonoYPixel;
struct ConstMonoYPixel;
struct RGBSolidPixel;
struct ConstRGBSolidPixel;
struct RGB210Pixel;
struct ConstRGB210Pixel;
struct RGBAPixel;
struct ConstRGBAPixel;
struct RGBASolidPixel;
struct ConstRGBASolidPixel;
struct RGBA64Pixel;
struct ConstRGBA64Pixel;

template <>
inline std::string typeName<MonoYPixel>(void) {
  return "MonoY";
}

template <>
inline std::string typeName<RGBSolidPixel>(void) {
  return "RGBSolid";
}
template <>
inline std::string typeName<ConstRGBSolidPixel>(void) {
  return "RGBSolid";
}

template <>
inline std::string typeName<RGB210Pixel>(void) {
  return "RGB210";
}
template <>
inline std::string typeName<ConstRGB210Pixel>(void) {
  return "RGB210";
}

template <>
inline std::string typeName<RGBAPixel>(void) {
  return "RGBA";
}
template <>
inline std::string typeName<ConstRGBAPixel>(void) {
  return "RGBA";
}

template <>
inline std::string typeName<RGBASolidPixel>(void) {
  return "RGBASolid";
}
template <>
inline std::string typeName<ConstRGBASolidPixel>(void) {
  return "RGBASolid";
}

template <>
inline std::string typeName<RGBA64Pixel>(void) {
  return "RGBA64";
}
/**
 * Tag for an 8bits grayscale pixel.
 */
struct MonoYPixel {
  typedef unsigned char value_type;
  typedef unsigned char buffer_value_type;
  typedef MonoYPixel NonConst;
  typedef ConstMonoYPixel Const;
};

/**
 * Tag for an 8bits grayscale pixel.
 */
struct ConstMonoYPixel {
  typedef const unsigned char value_type;
  typedef const unsigned char buffer_value_type;
  typedef MonoYPixel NonConst;
  typedef ConstMonoYPixel Const;
};

/**
 * Tag for a 24bits solid (no alpha) RGB pixel.
 */
struct RGBSolidPixel {
  typedef uchar3 value_type;
  typedef uchar3 buffer_value_type;
  typedef RGBSolidPixel NonConst;
  typedef ConstRGBSolidPixel Const;
};

/**
 * Tag for a 24bits solid (no alpha) RGB pixel.
 */
struct ConstRGBSolidPixel {
  typedef const uchar3 value_type;
  typedef const uchar3 buffer_value_type;
  typedef RGBSolidPixel NonConst;
  typedef ConstRGBSolidPixel Const;
};

/**
 * Tag for a 32bits-packed RGB210 pixel.
 * This has alpha on one bit and each color on 9 bits.
 */
struct RGB210Pixel {
  typedef uint32_t value_type;
  typedef uint32_t buffer_value_type;
  typedef RGB210Pixel NonConst;
  typedef ConstRGB210Pixel Const;
};

/**
 * Tag for a const 32bits-packed RGB210 pixel.
 */
struct ConstRGB210Pixel {
  typedef const uint32_t value_type;
  typedef const uint32_t buffer_value_type;
  typedef RGB210Pixel NonConst;
  typedef ConstRGB210Pixel Const;
};

/**
 * Tag for a 32bits RGBA pixel. This has each component on 8 bits.
 */
struct RGBAPixel {
  typedef uint32_t value_type;
  typedef uint32_t buffer_value_type;
  typedef RGBAPixel NonConst;
  typedef ConstRGBAPixel Const;
};

/**
 * Tag for a const 32bits RGBA pixel.
 */
struct ConstRGBAPixel {
  typedef const uint32_t value_type;
  typedef const uint32_t buffer_value_type;
  typedef RGBAPixel NonConst;
  typedef ConstRGBAPixel Const;
};

/**
 * Tag for a 32bits RGBA pixel. This has each component on 8 bits.
 */
struct RGBASolidPixel {
  typedef uint32_t value_type;
  typedef uint32_t buffer_value_type;
  typedef RGBASolidPixel NonConst;
  typedef ConstRGBASolidPixel Const;
};

/**
 * Tag for a const 32bits RGBA pixel.
 */
struct ConstRGBASolidPixel {
  typedef const uint32_t value_type;
  typedef const uint32_t buffer_value_type;
  typedef RGBASolidPixel NonConst;
  typedef ConstRGBASolidPixel Const;
};

/**
 * Tag for a 64bits RGBA pixel. This has each component on 16 bits.
 */
struct RGBA64Pixel {
  typedef uint64_t value_type;
  typedef uint64_t buffer_value_type;
  typedef RGBA64Pixel NonConst;
  typedef ConstRGBA64Pixel Const;
};

/**
 * Tag for a const 64bits RGBA pixel.
 */
struct ConstRGBA64Pixel {
  typedef const uint64_t value_type;
  typedef const uint64_t buffer_value_type;
  typedef RGBA64Pixel NonConst;
  typedef ConstRGBA64Pixel Const;
};

/**
 * The arrays don't own the data, they are just wrappers to type a device buffer.
 */
template <class PixelTag>
struct PixelArray {
 public:
  typedef PixelTag PixelT;
  typedef typename PixelTag::value_type value_type;
  typedef typename PixelTag::buffer_value_type buffer_value_type;

  /**
   * Builds an array.
   * @param width array width in pixels
   * @param height array height in pixels
   * @param buffer array data, of size width * height elements. Ownership is not taken.
   */
  PixelArray(int64_t width, int64_t height, GPU::Buffer<buffer_value_type> buffer)
      : width(width), height(height), buffer(buffer) {}

  //  PixelArray(int64_t width, int64_t height, buffer_value_type *buffer)
  //    : width(width), height(height), buffer(GPU::Buffer<buffer_value_type>(buffer, width * height *
  //    sizeof(buffer_value_type))) {}

  /**
   * Const pixel arrays can be converted fron non-const ones.
   */
  PixelArray(const PixelArray<typename PixelTag::NonConst>& other)
      : width(other.width), height(other.height), buffer(other.buffer) {}

  /**
   * Returns a const version of this array;
   */
  PixelArray<typename PixelTag::Const> constify() const { return PixelArray<typename PixelTag::Const>(*this); }

  /**
   * The width of the array.
   */
  const int64_t width;
  /**
   * The height of the array.
   */
  const int64_t height;
  /**
   * The data of the array.
   */
  GPU::Buffer<buffer_value_type> buffer;
};
}  // namespace Image
}  // namespace VideoStitch

#endif
