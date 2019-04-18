// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rect.hpp"

#include <algorithm>

namespace VideoStitch {
namespace Core {

void Rect::growToAlignTo(int x, int y) {
  if (empty()) {
    return;
  }
  left_ = (left_ / x) * x;
  top_ = (top_ / y) * y;
}

void Rect::growToMultipleSizeOf(int x, int y) {
  if (empty()) {
    return;
  }
  if (getWidth() % x) {
    right_ += x - (getWidth() % x);
  }
  if (getHeight() % y) {
    bottom_ += y - (getHeight() % y);
  }
  assert(getWidth() % x == 0);
  assert(getHeight() % y == 0);
}

void Rect::getInterAndUnion(const Rect &a, const Rect &b, Rect &iRect, Rect &uRect, unsigned wrapWidth) {
  uRect.top_ = std::min(a.top_, b.top_);
  uRect.bottom_ = std::max(a.bottom_, b.bottom_);
  uRect.left_ = std::min(a.left_, b.left_);
  uRect.right_ = std::max(a.right_, b.right_);

  // Need to handle the following cases
  if (std::max(a.right_, b.right_) >= wrapWidth && std::min(a.right_, b.right_) < wrapWidth) {
    /**
     *  *-------------------------*
     *  |    *-------------*      |
     *  |----+---*         |  *- -|
     *  |    |   |         |  |   |
     *  |    |   |         |  |   |
     *  |    *---+---------*  |   |
     *  |--------*            *---|
     *  |                         |
     *  *-------------------------*
     **/
    if (a.right() % wrapWidth < std::max(a.left_, b.left_) && b.right() % wrapWidth < std::max(a.left_, b.left_) &&
        std::min(a.left_, b.left_) <= std::max(a.right_, b.right_) - wrapWidth) {
      iRect.top_ = std::max(a.top_, b.top_);
      iRect.bottom_ = std::min(a.bottom_, b.bottom_);
      iRect.left_ = std::min(a.left_, b.left_);
      iRect.right_ = std::max(a.right_, b.right_) - wrapWidth;

      uRect.top_ = std::min(a.top_, b.top_);
      uRect.bottom_ = std::max(a.bottom_, b.bottom_);
      uRect.left_ = std::max(a.left_, b.left_);
      uRect.right_ = std::min(a.right_, b.right_) + wrapWidth;
      return;
    } else {
      /**
       *  *-------------------------*
       *  |    *-------------*      |
       *  |--* |         *---|------|
       *  |  | |         |   |      |
       *  |  | |         |   |      |
       *  |  | *---------|---*      |
       *  |--*           *----------|
       *  |                         |
       *  *-------------------------*
       **/
      if (std::max(a.left_, b.left_) < std::min(a.right_, b.right_) &&
          std::min(a.left_, b.left_) > std::max(a.right_, b.right_) - wrapWidth) {
        iRect.top_ = std::max(a.top_, b.top_);
        iRect.bottom_ = std::min(a.bottom_, b.bottom_);
        iRect.left_ = std::max(a.left_, b.left_);
        iRect.right_ = std::min(a.right_, b.right_);
        return;
      }
    }
  }

  /**
   * We must handle the following case:
   *
   *  *-------------------------*
   *  |    *----------------*   |
   *  |----+---*          *-+---|
   *  |    |   |          | |   |
   *  |    |   |          | |   |
   *  |    *---+----------+-*   |
   *  |--------*          *-----|
   *  |                         |
   *  *-------------------------*
   *
   * The intersection will be the smallest area rectangle of the two possible rectangles.
   *
   * the intersections are either (in order):
   *   - [a.left, b.right] and [b.left, a.right], in which case the two possibilities are [a.left, a.right] and [b.left,
   * b.right].
   *   - [b.left, a.right] and [a.left, b.right], in which case the two possibilities are [b.left, b.right] and [a.left,
   * a.right].
   */
  iRect.top_ = std::max(a.top_, b.top_);
  iRect.bottom_ = std::min(a.bottom_, b.bottom_);
  if (uRect.right_ >= (int)wrapWidth && uRect.right_ - (int)wrapWidth >= uRect.left_) {
    if (a.getWidth() < b.getWidth()) {
      iRect.left_ = a.left_;
      iRect.right_ = a.right_;
    } else {
      iRect.left_ = b.left_;
      iRect.right_ = b.right_;
    }
    // uRect's anchor is arbitrary, but it must contain iRetc
    if (iRect.right_ >= (int)wrapWidth) {
      // make uRect wrap, choose to center iRect within uRect
      uRect.left_ = iRect.left_ - (int)(wrapWidth - iRect.getWidth()) / 2;
      if (uRect.left_ < 0) {
        uRect.left_ = 0;
      }
      uRect.right_ = uRect.left_ + (int)wrapWidth - 1;
    } else {
      // make uRect NOT wrap
      uRect.left_ = 0;
      uRect.right_ = (int)wrapWidth - 1;
    }
  } else {
    iRect.left_ = std::max(a.left_, b.left_);
    iRect.right_ = std::min(a.right_, b.right_);
  }
}

}  // namespace Core
}  // namespace VideoStitch
