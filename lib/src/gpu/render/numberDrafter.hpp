// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {
namespace Render {

/**
 * @brief A class to draw numbers in CUDA buffers.
 */
class NumberDrafter {
 public:
  /**
   * Create a NumberDrafter.
   * @param width The width of numbers, in fractional pixels.
   */
  explicit NumberDrafter(float width);

  /**
   * Returns the height of numbers, in fractional pixels.
   */
  float getNumberHeight() const;

  /**
   * Returns the height of numbers for the given width, in fractional pixels.
   */
  static float getNumberHeightForWidth(float width);

  /**
   * Returns the width of numbers for the given height, in fractional pixels.
   */
  static float getNumberWidthForHeight(float height);

  /**
   * Draw a specific digit.
   * @param digit Digit to draw. Must be in {0,1,2,3,4,5,6,7,8,9}.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw(int digit, Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
            GPU::Stream stream) const;

  /**
   * Draw a 0.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw0(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 1.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw1(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 2.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw2(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 3.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw3(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 4.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw4(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 5.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw5(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 6.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw6(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 7.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw7(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw an 8.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw8(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

  /**
   * Draw a 9.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param x left of the place where to write the number.
   * @param y top of the place where to write the number.
   * @param color Strike color.
   * @param stream Cuda Sstream where to run the kernels.
   * @note Asynchronous.
   */
  template <typename Image>
  void draw9(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
             GPU::Stream stream) const;

 private:
  const float width;
  const float thickness;
  // These are the values that are used by several numbers, relative to the bounding box of the numbers.
  const float centersX;
  const float upperCenterY;
  const float lowerCenterY;
  const float leftLineX;
  const float rightLineX;
  const float topLineY;
  const float bottomLineY;
  const float innerSqrRadius;
  const float outerSqrRadius;
};
}  // namespace Render
}  // namespace VideoStitch
