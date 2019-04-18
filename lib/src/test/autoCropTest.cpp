// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/imageProcessingUtils.hpp"

#include <cassert>
#include <iostream>
#include <memory>

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "../util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Testing {

void testCircle() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif

  std::vector<std::string> autoCropTests;

  for (int i = 0; i < 1; i++) {
    autoCropTests.push_back(workingPath + "data/autocrop/test" + std::to_string(i) + ".png");
  }
  std::vector<int> circleXs = {237};
  std::vector<int> circleYs = {177};
  std::vector<int> circleRadiuses = {260};

  for (size_t i = 0; i < autoCropTests.size(); i++) {
    int channelCount;
    int64_t width, height;
    std::vector<unsigned char> data;
    int x, y, radius;
    if (!Util::ImageProcessing::readImage(autoCropTests[i], width, height, channelCount, data).ok()) {
      std::cout << "Test " << i << ": Cannot read " << autoCropTests[i] << std::endl;
      ENSURE_EQ(1, 0);
    }
#if defined(DUMP_TEST_RESULT)
    Status status = Util::ImageProcessing::findCropCircle((int)width, (int)height, &data[0], x, y, radius, nullptr,
                                                          *autoCropTests[i]);
    if (!status) {
      std::cout << "Test " << i << ": " << status.getErrorMessage() << std::endl;
    }
#else
    Status status = Util::ImageProcessing::findCropCircle((int)width, (int)height, &data[0], x, y, radius);
    if (!status.ok()) {
      std::cout << "Test " << i << ": " << status.getErrorMessage() << std::endl;
    }
    std::cout << "Test " << i << ": " << x << " " << y << " " << radius << std::endl;
    ENSURE((circleXs[i] - x) <= 2, "x differs more than 2 pixels");
    ENSURE((circleYs[i] - y) <= 2, "y differs more than 2 pixels");
    ENSURE((circleRadiuses[i] - radius) <= 2, "r differs more than 2 pixels");
#endif
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::testCircle();

  return 0;
}
