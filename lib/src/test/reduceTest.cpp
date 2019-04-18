// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Poisson solver tests.

#include "gpu/testing.hpp"
#include "gpu/util.hpp"
#include <gpu/image/reduce.hpp>
#include "libvideostitch/gpu_device.hpp"
#include <vector>

namespace VideoStitch {
namespace Testing {
void testReduceSum(const int size) {
  std::vector<uint32_t> inputData;
  uint32_t expectedSum = 0;
  for (uint32_t i = 0; (int)i < size; ++i) {
    const uint32_t val = (i * i) % 4786;
    expectedSum += val;
    inputData.push_back(val);
  }

  DeviceBuffer<uint32_t> buffer(size, 1, inputData.data());
  DeviceBuffer<uint32_t> work(Image::getReduceWorkBufferSize(size), 1);

  uint32_t sum = 0;
  ENSURE(Image::reduceSum(buffer.gpuBuf(), work.gpuBuf(), size, sum));

  ENSURE_EQ(expectedSum, sum);
}

void testReduceSolid(const int size) {
  std::vector<uint32_t> inputData;
  uint32_t expectedCount = 0;
  for (uint32_t i = 0; (int)i < size; ++i) {
    const int alpha = ((i * 154789) % 255) > 128 ? 1 : 0;
    const uint32_t val = Image::RGBA::pack(1, 1, 1, alpha);
    if (alpha) {
      expectedCount += 1;
    }
    inputData.push_back(val);
  }

  // Count test
  {
    DeviceBuffer<uint32_t> buffer(size, 1, inputData.data());
    DeviceBuffer<uint32_t> work(Image::getReduceWorkBufferSize(size), 1);

    uint32_t count = 0;
    ENSURE(Image::reduceCountSolid(buffer.gpuBuf(), work.gpuBuf(), size, count));

    ENSURE_EQ(expectedCount, count);
  }

  // Sum test
  {
    DeviceBuffer<uint32_t> buffer(size, 1, inputData.data());
    DeviceBuffer<uint32_t> work(Image::getReduceWorkBufferSize(size), 1);

    uint32_t sum = 0;
    ENSURE(Image::reduceSumSolid(buffer.gpuBuf(), work.gpuBuf(), size, sum));

    ENSURE_EQ(expectedCount, sum);
  }
}
}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testReduceSum(3);
  VideoStitch::Testing::testReduceSum(257);
  VideoStitch::Testing::testReduceSum(513);
  VideoStitch::Testing::testReduceSum(1234567);
  VideoStitch::Testing::testReduceSolid(3);
  VideoStitch::Testing::testReduceSolid(257);
  VideoStitch::Testing::testReduceSolid(513);
  VideoStitch::Testing::testReduceSolid(1234567);
  return 0;
}
