// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

/**
 * Grid preprocessor tests
 */

#include "gpu/testing.hpp"
#include "common/ptv.hpp"
#include "gpu/util.hpp"

#include "libvideostitch/context.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/logging.hpp"

#include <gpu/memcpy.hpp>
#include <gpu/processors/grid.hpp>

#include <memory>
#include <sstream>
#include <vector>

namespace VideoStitch {
namespace Testing {

void golden_grid(uint32_t* dst, unsigned width, unsigned height, unsigned size, unsigned lineWidth, uint32_t color,
                 uint32_t bgColor) {
  for (unsigned x = 0; x < width; ++x) {
    for (unsigned y = 0; y < height; ++y) {
      dst[y * width + x] = ((x % size) < lineWidth || (y % size) < lineWidth) ? color : bgColor;
    }
  }
}

void golden_transparentBGGridKernel(uint32_t* dst, unsigned width, unsigned height, unsigned size, unsigned lineWidth,
                                    uint32_t color) {
  for (unsigned x = 0; x < width; ++x) {
    for (unsigned y = 0; y < height; y++) {
      if ((x % size) < lineWidth || (y % size) < lineWidth) {
        dst[y * width + x] = color;
      }
    }
  }
}

void golden_transparentFGGridKernel(uint32_t* dst, unsigned width, unsigned height, unsigned size, unsigned lineWidth,
                                    uint32_t bgColor) {
  for (unsigned x = 0; x < width; ++x) {
    for (unsigned y = 0; y < height; y++) {
      if (!((x % size) < lineWidth || (y % size) < lineWidth)) {
        dst[y * width + x] = bgColor;
      }
    }
  }
}

void testGridPreProcessor(unsigned width, unsigned height, unsigned size, unsigned lineWidth, uint32_t bgColor,
                          uint32_t color) {
  auto stream = createTestedStream();
  auto devGrid = GPU::Buffer<uint32_t>::allocate(width * height, "testGrid");
  ENSURE(devGrid.status());
  auto devGridFG = GPU::Buffer<uint32_t>::allocate(width * height, "testGrid");
  ENSURE(devGridFG.status());
  auto devGridBG = GPU::Buffer<uint32_t>::allocate(width * height, "testGrid");
  ENSURE(devGridBG.status());

  std::vector<uint32_t> actualGrid(width * height);
  std::vector<uint32_t> actualGridFG(width * height);
  std::vector<uint32_t> actualGridBG(width * height);

  ENSURE(memsetToZeroAsync(devGrid.value(), stream));
  ENSURE(memsetToZeroAsync(devGridBG.value(), stream));
  ENSURE(memsetToZeroAsync(devGridFG.value(), stream));

  Core::grid(devGrid.value(), width, height, size, lineWidth, color, bgColor, stream);
  Core::transparentForegroundGrid(devGridFG.value(), width, height, size, lineWidth, bgColor, stream);
  Core::transparentBackgroundGrid(devGridBG.value(), width, height, size, lineWidth, color, stream);

  std::vector<uint32_t> expectedGrid(width * height);
  std::vector<uint32_t> expectedGridBG(width * height);
  std::vector<uint32_t> expectedGridFG(width * height);

  golden_grid(expectedGrid.data(), width, height, size, lineWidth, color, bgColor);
  golden_transparentBGGridKernel(expectedGridBG.data(), width, height, size, lineWidth, color);
  golden_transparentFGGridKernel(expectedGridFG.data(), width, height, size, lineWidth, bgColor);

  ENSURE(GPU::memcpyAsync(actualGrid.data(), (GPU::Buffer<const uint32_t>)devGrid.value(), stream));
  ENSURE(GPU::memcpyAsync(actualGridFG.data(), (GPU::Buffer<const uint32_t>)devGridFG.value(), stream));
  ENSURE(GPU::memcpyAsync(actualGridBG.data(), (GPU::Buffer<const uint32_t>)devGridBG.value(), stream));

  stream.synchronize();

  ENSURE_RGBA210_ARRAY_EQ(expectedGrid.data(), actualGrid.data(), width, height);
  ENSURE_RGBA210_ARRAY_EQ(expectedGridBG.data(), actualGridBG.data(), width, height);
  ENSURE_RGBA210_ARRAY_EQ(expectedGridFG.data(), actualGridFG.data(), width, height);

  ENSURE(devGrid.value().release());
  ENSURE(devGridFG.value().release());
  ENSURE(devGridBG.value().release());
  stream.destroy();
}
}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  uint32_t color = 0xff0000ff;  // ABGR
  uint32_t bgColor = 0xff000000;
  const int size = 32;
  const int lineWidth = 2;
  VideoStitch::Testing::testGridPreProcessor(153, 234, size, lineWidth, bgColor, color);
  VideoStitch::GPU::Context::destroy();
  return 0;
}
