// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// This file defines the host and device utilities for representing pixel arrays.
//

#ifndef COLOR_ARRAY_DEVICE_HPP_
#define COLOR_ARRAY_DEVICE_HPP_

// This part of the code is available only to CUDA.
#ifndef __CUDACC__
#error "You included this file in non-device code. This should not happen."
#endif

#include "backend/common/imageOps.hpp"

#include "image/colorArray.hpp"

#include <cassert>

#ifdef __linux__
#define RESTRICT_MEMBER __restrict__
#else
#define RESTRICT_MEMBER
#endif

namespace VideoStitch {
namespace Image {

namespace Internal {

/**
 * Accessor for "typical" row-major arrays, i.e. arrays where pixels are given sequentially each on sizeof(value_type)
 * bytes.
 */
template <typename PixelTag>
struct RowMajorImpl {
  static __device__ typename PixelTag::value_type& at(int i, typename PixelTag::buffer_value_type* buffer) {
    return buffer[i];
  }
};

/**
 * Default impl does nothing.
 */
template <typename PixelTag>
struct Impl {};

/**
 * Specializations for all pixel tags.
 * @{
 */
template <>
struct Impl<MonoYPixel> : RowMajorImpl<MonoYPixel> {
  typedef MonoY8 GetterT;
};

template <>
struct Impl<ConstMonoYPixel> : RowMajorImpl<ConstMonoYPixel> {
  typedef MonoY8 GetterT;
};

template <>
struct Impl<RGBSolidPixel> : RowMajorImpl<RGBSolidPixel> {
  typedef RGBSolid GetterT;
};

template <>
struct Impl<ConstRGBSolidPixel> : RowMajorImpl<ConstRGBSolidPixel> {
  typedef RGBSolid GetterT;
};

template <>
struct Impl<RGB210Pixel> : RowMajorImpl<RGB210Pixel> {
  typedef RGB210 GetterT;
};

template <>
struct Impl<ConstRGB210Pixel> : RowMajorImpl<ConstRGB210Pixel> {
  typedef RGB210 GetterT;
};

template <>
struct Impl<RGBAPixel> : RowMajorImpl<RGBAPixel> {
  typedef RGBA GetterT;
};

template <>
struct Impl<ConstRGBAPixel> : RowMajorImpl<ConstRGBAPixel> {
  typedef RGBA GetterT;
};

template <>
struct Impl<RGBASolidPixel> : RowMajorImpl<RGBASolidPixel> {
  typedef RGBASolid GetterT;
};

template <>
struct Impl<ConstRGBASolidPixel> : RowMajorImpl<ConstRGBASolidPixel> {
  typedef RGBASolid GetterT;
};

template <>
struct Impl<RGBA64Pixel> : RowMajorImpl<RGBA64Pixel> {
  typedef RGBA64 GetterT;
};

template <>
struct Impl<ConstRGBA64Pixel> : RowMajorImpl<ConstRGBA64Pixel> {
  typedef RGBA64 GetterT;
};

/**
 * @}
 */
}  // namespace Internal

/**
 * A KernelPixelArray is basically a PixelArray augmented with what's useful for kernels to have,
 * i.e. tools to manipulate the pixels (get the pixel at x,y and break into individual color components).
 */
template <typename PixelTag>
class KernelPixelArray {
 public:
  typedef PixelTag PixelT;
  typedef typename PixelTag::value_type value_type;
  typedef typename PixelTag::buffer_value_type buffer_value_type;
  typedef typename Internal::Impl<PixelTag>::GetterT GetterT;

  explicit KernelPixelArray(const PixelArray<PixelTag>& wrapped)
      : width((int)wrapped.width), height((int)wrapped.height), buffer(wrapped.buffer.get()) {}

  /**
   * Returns the pixel at coords (@a x, @a y).
   * No bounds checking.
   */
  inline __device__ value_type& at(int x, int y) const { return at(y * width + x); }

  /**
   * Returns the pixel at row-major coord (@a i).
   * No bounds checking.
   */
  inline __device__ value_type& at(int i) const { return Internal::Impl<PixelTag>::at(i, buffer); }

  /**
   * The width of the array.
   */
  const int width;
  /**
   * The height of the array.
   */
  const int height;

 private:
  /**
   * The data of the array.
   */
  buffer_value_type* const buffer;
};

/**
 * An KernelInOutPixelArray is basically two KernelPixelArray's (an input and output) that have the same size.
 */
template <typename InPixelTag, typename OutPixelTag>
class KernelInOutPixelArray {
 public:
  typedef InPixelTag InPixelT;
  typedef OutPixelTag OutPixelT;
  typedef typename InPixelTag::value_type in_value_type;
  typedef typename OutPixelTag::value_type out_value_type;
  typedef typename InPixelTag::buffer_value_type in_buffer_value_type;
  typedef typename OutPixelTag::buffer_value_type out_buffer_value_type;
  typedef typename Internal::Impl<InPixelTag>::GetterT InGetterT;
  typedef typename Internal::Impl<OutPixelTag>::GetterT OutGetterT;

  explicit KernelInOutPixelArray(const PixelArray<InPixelTag>& input, const PixelArray<OutPixelTag>& output)
      : width((int)input.width),
        height((int)input.height),
        inBuffer(input.buffer.get()),
        outBuffer(output.buffer.get()) {
    assert(input.width == output.width);
    assert(input.height == output.height);
  }

  /**
   * Returns the input pixel at coords (@a x, @a y).
   * No bounds checking.
   */
  inline __device__ in_value_type& inAt(int x, int y) const { return inAt(y * width + x); }

  /**
   * Returns the input pixel at row-major coord (@a i).
   * No bounds checking.
   */
  inline __device__ in_value_type& inAt(int i) const { return Internal::Impl<InPixelTag>::at(i, inBuffer); }

  /**
   * Returns the output pixel at coords (@a x, @a y).
   * No bounds checking.
   */
  inline __device__ out_value_type& outAt(int x, int y) const { return outAt(y * width + x); }

  /**
   * Returns the output pixel at row-major coord (@a i).
   * No bounds checking.
   */
  inline __device__ out_value_type& outAt(int i) const { return Internal::Impl<OutPixelTag>::at(i, outBuffer); }

  /**
   * The width of the array.
   */
  const int width;
  /**
   * The height of the array.
   */
  const int height;

 private:
  /**
   * The data of the array.
   */
  RESTRICT_MEMBER in_buffer_value_type* const inBuffer;
  RESTRICT_MEMBER out_buffer_value_type* const outBuffer;
};

/**
 * A helper function for automatic type inference.
 */
template <typename PixelTag>
KernelPixelArray<PixelTag> wrapForKernel(const PixelArray<PixelTag>& array) {
  return KernelPixelArray<PixelTag>(array);
}

/**
 * A helper function for automatic type inference.
 */
template <typename InPixelTag, typename OutPixelTag>
KernelInOutPixelArray<InPixelTag, OutPixelTag> wrapForKernel(const PixelArray<InPixelTag>& input,
                                                             const PixelArray<OutPixelTag>& output) {
  return KernelInOutPixelArray<InPixelTag, OutPixelTag>(input, output);
}
}  // namespace Image
}  // namespace VideoStitch

#endif
