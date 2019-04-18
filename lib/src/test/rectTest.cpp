// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include <core/rect.hpp>

#include <vector>

namespace VideoStitch {
namespace Testing {

inline Core::Rect createRect(int64_t top, int64_t left, int64_t bottom, int64_t right) {
  return Core::Rect::fromInclusiveTopLeftBottomRight(top, left, bottom, right);
}

void testRectInit() {
  Core::Rect rect;
  ENSURE(rect.empty());
  ENSURE(rect.verticallyEmpty());
  ENSURE(rect.horizontallyEmpty());
  ENSURE_EQ(rect.getArea(), (int64_t)0);
  ENSURE(rect == Core::Rect{});
}

void testRectInterAndUnion() {
  // Test case 1
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
  // Test case 2
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
  // Test case 3 : Not handle correctly in current implementation. Need to fix this later
  /**
   *  *-------------------------*
   *  |    *----------------*   |
   *  |----+---*          *-+---|
   *  |    |   |          | |   |
   *  |    |   |          | |   |
   *  |    *---+----------+-*   |
   *  |--------*          *-----|
   *  |                         |
   *  *-------------------------*
   **/
  std::vector<Core::Rect> rect0s = {createRect(1, 5, 7, 12), createRect(1, 5, 7, 12), createRect(1, 5, 7, 12)};
  std::vector<Core::Rect> rect1s = {createRect(3, 1, 5, 3), createRect(5, 4, 9, 8), createRect(1, 1, 2, 7)};

  std::vector<Core::Rect> iRects = {createRect(3, 1, 5, 2), createRect(5, 5, 7, 8), createRect(1, 1, 2, 2)};
  std::vector<Core::Rect> uRects = {createRect(1, 5, 7, 13), createRect(1, 4, 9, 12), createRect(1, 0, 7, 9)};

  for (size_t i = 0; i < 2; i++) {
    Core::Rect iRect = createRect(0, 0, 0, 0);
    Core::Rect uRect = createRect(0, 0, 0, 0);
    Core::Rect::getInterAndUnion(rect0s[i], rect1s[i], iRect, uRect, 10);

    ENSURE_EQ(iRect.left(), iRects[i].left());
    ENSURE_EQ(iRect.right(), iRects[i].right());
    ENSURE_EQ(iRect.top(), iRects[i].top());
    ENSURE_EQ(iRect.bottom(), iRects[i].bottom());

    ENSURE_EQ(uRect.left(), uRects[i].left());
    ENSURE_EQ(uRect.right(), uRects[i].right());
    ENSURE_EQ(uRect.top(), uRects[i].top());
    ENSURE_EQ(uRect.bottom(), uRects[i].bottom());
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int, char **) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testRectInit();
  VideoStitch::Testing::testRectInterAndUnion();

  return 0;
}
