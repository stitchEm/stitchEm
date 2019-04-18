// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

namespace VideoStitch {
namespace Util {

static const double ASPECT_RATIO(2.0);
static const int MIN_PANO_WIDTH(4);
static const int MIN_PANO_HEIGHT(2);

/**
 * @brief Represents a panorama size.
 */
struct PanoSize {
  PanoSize() : width(0), height(0) {}
  PanoSize(const unsigned w, const unsigned h) : width(w), height(h) {}
  int width;
  int height;
};

/**
 * @brief Given the height and the aspect ratio, calculates the width.
 * @param hValue The height value.
 * @return The new width value.
 */
static unsigned calculateWidth(const int hValue) {
  if (hValue % 2 != 0) {
    return (hValue + 1) * ASPECT_RATIO;
  } else {
    return hValue * ASPECT_RATIO;
  }
}

/**
 * @brief Given the width and the aspect ratio, calculates the height.
 * @param wValue The width value.
 * @return The new height value.
 */
static unsigned calculateHeight(const int wValue) {
  unsigned h = wValue / ASPECT_RATIO;
  if (h % 2 == 0) {
    return h;
  } else {
    return ++h;
  }
}

/**
 * @brief Given the width, calculates the height and adjusts the width.
 * @param width The width.
 * @return The adjusted size.
 */
static inline PanoSize calculateSizeFromWidth(const int width) {
  if (width < MIN_PANO_WIDTH) {
    return PanoSize(MIN_PANO_WIDTH, MIN_PANO_HEIGHT);
  }
  const unsigned valueH = VideoStitch::Util::calculateHeight(width);
  const unsigned valueW = VideoStitch::Util::calculateWidth(valueH);
  return PanoSize(valueW, valueH);
}

/**
 * @brief Given the height, calculates the width and adjusts the height.
 * @param height The height.
 * @return The adjusted size.
 */
static inline PanoSize calculateSizeFromHeight(const int height) {
  if (height < MIN_PANO_HEIGHT) {
    return PanoSize(MIN_PANO_WIDTH, MIN_PANO_HEIGHT);
  }
  const unsigned valueW = VideoStitch::Util::calculateWidth(height);
  const unsigned valueH = VideoStitch::Util::calculateHeight(valueW);
  return PanoSize(valueW, valueH);
}
}  // namespace Util
}  // namespace VideoStitch
