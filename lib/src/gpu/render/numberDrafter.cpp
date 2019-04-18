// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/render/numberDrafter.hpp"

#include "geometry.hpp"

namespace VideoStitch {
namespace Render {

namespace {
/**
 * Return the thickness from a given width.
 */
float thicknessFromWidth(float width) {
  // Make sure to update getNumberWidthForHeight if you change that.
  return width / 4.0f;
}
}  // namespace

float NumberDrafter::getNumberWidthForHeight(float height) {
  // Make sure to update thicknessFromWidth if you change that.
  return 8.0f * height / 15.0f;
}

NumberDrafter::NumberDrafter(float width)
    : width(width),
      thickness(thicknessFromWidth(width)),
      centersX(width / 2.0f),
      upperCenterY(width / 2.0f),
      lowerCenterY(upperCenterY + width - thickness / 2.0f),
      leftLineX(thickness / 4.0f),
      rightLineX(width - thickness / 4.0f),
      topLineY(thickness / 4.0f),
      bottomLineY(2.0f * width - thickness / 2.0f - thickness / 4.0f),
      innerSqrRadius(((width - thickness) * (width - thickness)) / 4.0f),
      outerSqrRadius((width * width) / 4.0f) {}

float NumberDrafter::getNumberHeight() const { return 2.0f * width - thickness / 2.0f; }

float NumberDrafter::getNumberHeightForWidth(float width) { return 2.0f * width - thicknessFromWidth(width) / 2.0f; }

template <typename Image>
void NumberDrafter::draw(int digit, Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                         GPU::Stream stream) const {
  switch (digit) {
    case 0:
      draw0(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 1:
      draw1(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 2:
      draw2(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 3:
      draw3(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 4:
      draw4(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 5:
      draw5(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 6:
      draw6(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 7:
      draw7(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 8:
      draw8(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    case 9:
      draw9(dst, dstWidth, dstHeight, x, y, color, stream);
      return;
    default:
      return;
  }
}

template <typename Image>
void NumberDrafter::draw0(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircleTop(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color,
                stream);
  drawCircleBottom(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color,
                   stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + upperCenterY, x + leftLineX, y + lowerCenterY,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + rightLineX, y + upperCenterY, x + rightLineX, y + lowerCenterY,
           thickness * thickness / 4.0f, color, stream);
}

template <typename Image>
void NumberDrafter::draw1(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawLine(dst, dstWidth, dstHeight, x + 0.5f * width, y + topLineY, x + 0.5f * width, y + bottomLineY,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + 0.25f * width, y + bottomLineY, x + 0.75f * width, y + bottomLineY,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + 0.5f * width, y + topLineY, x + 0.12f * width, y + 0.12f * width,
           thickness * thickness / 4.0f, color, stream);
}

template <typename Image>
void NumberDrafter::draw2(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircleTop(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color,
                stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + bottomLineY, x + rightLineX, y + bottomLineY,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + bottomLineY, x + rightLineX, y + upperCenterY,
           thickness * thickness / 4.0f, color, stream);
}

template <typename Image>
void NumberDrafter::draw3(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircleTop(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color,
                stream);
  drawCircleBottomRight(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color,
                        stream);
  drawCircleBottom(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color,
                   stream);
  drawCircleTopRight(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color,
                     stream);
}

template <typename Image>
void NumberDrafter::draw4(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  float tt = 0.66f * getNumberHeight();
  drawLine(dst, dstWidth, dstHeight, x + 0.75f * width, y + topLineY, x + 0.75f * width, y + bottomLineY,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + 0.75f * width, y + topLineY, x + leftLineX, y + tt,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + tt, x + rightLineX, y + tt, thickness * thickness / 4.0f, color,
           stream);
}

template <typename Image>
void NumberDrafter::draw5(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircleBottom(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color,
                   stream);
  drawCircleTopRight(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color,
                     stream);
  float mid = getNumberHeight() / 2.0f;
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + mid, x + centersX, y + mid, thickness * thickness / 4.0f, color,
           stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + mid, x + leftLineX, y + topLineY, thickness * thickness / 4.0f,
           color, stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + topLineY, x + 0.75f * width, y + topLineY,
           thickness * thickness / 4.0f, color, stream);
}

template <typename Image>
void NumberDrafter::draw6(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircleTop(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color,
                stream);
  drawCircle(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + leftLineX, y + upperCenterY, x + leftLineX, y + lowerCenterY,
           thickness * thickness / 4.0f, color, stream);
}

template <typename Image>
void NumberDrafter::draw7(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawLine(dst, dstWidth, dstHeight, x + 0.75f * width, y + topLineY, x + 0.25f * width, y + bottomLineY,
           thickness * thickness / 4.0f, color, stream);
  drawLine(dst, dstWidth, dstHeight, x + 0.12f * width, y + topLineY, x + 0.75f * width, y + topLineY,
           thickness * thickness / 4.0f, color, stream);
}

template <typename Image>
void NumberDrafter::draw8(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircle(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color, stream);
  drawCircle(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color, stream);
}

template <typename Image>
void NumberDrafter::draw9(Image& dst, int64_t dstWidth, int64_t dstHeight, float x, float y, uint32_t color,
                          GPU::Stream stream) const {
  drawCircle(dst, dstWidth, dstHeight, x + centersX, y + upperCenterY, innerSqrRadius, outerSqrRadius, color, stream);
  drawCircleBottom(dst, dstWidth, dstHeight, x + centersX, y + lowerCenterY, innerSqrRadius, outerSqrRadius, color,
                   stream);
  drawLine(dst, dstWidth, dstHeight, x + rightLineX, y + upperCenterY, x + rightLineX, y + lowerCenterY,
           thickness * thickness / 4.0f, color, stream);
}

template void NumberDrafter::draw(int digit, GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                  uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw0(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw1(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw2(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw3(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw4(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw5(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw6(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw7(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw8(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw9(GPU::Surface& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;

template void NumberDrafter::draw(int digit, GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x,
                                  float y, uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw0(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw1(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw2(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw3(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw4(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw5(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw6(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw7(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw8(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
template void NumberDrafter::draw9(GPU::Buffer<uint32_t>& dst, int64_t dstWidth, int64_t dstHeight, float x, float y,
                                   uint32_t color, GPU::Stream stream) const;
}  // namespace Render
}  // namespace VideoStitch
