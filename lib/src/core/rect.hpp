// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"

#include <cassert>

namespace VideoStitch {
namespace Core {
/**
 * @brief A simple Rectangle class.
 */
struct VS_EXPORT Rect {
  static Rect fromInclusiveTopLeftBottomRight(int64_t top, int64_t left, int64_t bottom, int64_t right) {
    return Rect{top, left, bottom, right};
  }

  Rect() : top_(0), left_(0), bottom_(-1), right_(-1) {}

  /**
   * Copy contructor.
   */
  Rect(const Rect& o) : top_(o.top_), left_(o.left_), bottom_(o.bottom_), right_(o.right_) {}

  /**
   * Returns the top coordinate of the rectangle.
   */
  int64_t top() const { return top_; }
  /**
   * Returns the left coordinate of the rectangle.
   */
  int64_t left() const { return left_; }
  /**
   * Returns the bottom coordinate of the rectangle.
   */
  int64_t bottom() const { return bottom_; }
  /**
   * Returns the right coordinate of the rectangle.
   */
  int64_t right() const { return right_; }
  /**
   * Set the top coordinate of the rectangle.
   */
  void setTop(int64_t top) { top_ = top; }
  /**
   * Set the left coordinate of the rectangle.
   */
  void setLeft(int64_t left) { left_ = left; }
  /**
   * Set the bottom coordinate of the rectangle.
   */
  void setBottom(int64_t bottom) { bottom_ = bottom; }
  /**
   * Set the right coordinate of the rectangle.
   */
  void setRight(int64_t right) { right_ = right; }
  /**
   * Returns the width of the rectangle.
   */
  int64_t getWidth() const {
    assert(right_ >= left_);
    return right_ - left_ + 1;
  }
  /**
   * Returns the height of the rectangle.
   */
  int64_t getHeight() const {
    assert(bottom_ >= top_);
    return bottom_ - top_ + 1;
  }
  /**
   * Returns true if the rectangle is empty vertically.
   */
  bool verticallyEmpty() const { return top_ > bottom_; }
  /**
   * Returns true if the rectangle is empty horizontally.
   */
  bool horizontallyEmpty() const { return left_ > right_; }
  /**
   * Returns true if the rectangle is empty (area <= 0).
   */
  bool empty() const { return horizontallyEmpty() || verticallyEmpty(); }
  /**
   * Returns the area of the rectangle.
   */
  int64_t getArea() const { return empty() ? 0 : getWidth() * getHeight(); }

  /**
   * Move left and top so that they are aligned on a multiple of the given dimensions.
   * They can only decrease. Right and bottom are not touched.
   */
  void growToAlignTo(int x, int y);

  /**
   * Move right and bottom so that the width and height are divisible by the given dimensions.
   * They can only increase. Top and left are not touched.
   */
  void growToMultipleSizeOf(int x, int y);

  /**
   * Fill iRect and uRect with the intersection and union of the two rectangles a and b.
   */
  static void getInterAndUnion(const Rect& a, const Rect& b, Rect& iRect, Rect& uRect, unsigned wrapWidth);

  /**
   * Returns true if rectangles are the same.
   */
  bool operator==(const Rect& o) const {
    return empty() == o.empty() || (top_ == o.top_ && left_ == o.left_ && bottom_ == o.bottom_ && right_ == o.right_);
  }

  Rect& operator=(const Rect& o) {
    top_ = o.top_;
    bottom_ = o.bottom_;
    right_ = o.right_;
    left_ = o.left_;
    return *this;
  }

  /**
   * Returns true if *this is inside o. An empty rectangle is inside all rectangles.
   */
  bool isInside(const Rect& o) const {
    if (empty()) {
      return true;
    } else if (o.empty()) {
      return false;
    } else {
      return (top_ >= o.top_ && left_ >= o.left_ && bottom_ <= o.bottom_ && right_ <= o.right_);
    }
  }

  /**
   * Returns true if point is contained in *this
   */
  bool contains(int x, int y) const { return left_ <= x && x <= right_ && top_ <= y && y <= bottom_; }

 private:
  /**
   * Constructor. Values are inclusive.
   */
  Rect(int64_t top, int64_t left, int64_t bottom, int64_t right)
      : top_(top), left_(left), bottom_(bottom), right_(right) {}

  int64_t top_;
  int64_t left_;
  int64_t bottom_;
  int64_t right_;
};
}  // namespace Core
}  // namespace VideoStitch
