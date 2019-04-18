// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "common/testing.hpp"
#include "common/shiftBuffers.hpp"

#include <core/pyramid.hpp>
#include "libvideostitch/gpu_device.hpp"
#include <backend/cuda/deviceBuffer.hpp>

namespace VideoStitch {
namespace Testing {
void testRoundTrip(const uint32_t* in, unsigned w, unsigned h, unsigned numLevels, bool inPlace) {
  uint32_t* devBuffer;
  ENSURE_CUDA(cudaMalloc((void**)&devBuffer, w * h * 4));
  ENSURE_CUDA(cudaMemcpy(devBuffer, in, w * h * 4, cudaMemcpyHostToDevice));

  Core::LaplacianPyramid<uint32_t>* pyr =
      Core::LaplacianPyramid<uint32_t>::create("test", w, h, numLevels,
                                               inPlace ? Core::LaplacianPyramid<uint32_t>::ExternalFirstLevel
                                                       : Core::LaplacianPyramid<uint32_t>::InternalFirstLevel,
                                               Core::LaplacianPyramid<uint32_t>::SingleShot,
                                               5 /*Gaussian filter radius*/, 2 /*Passes*/, false)
          .release();

  auto devGPUBuf = GPU::DeviceBuffer<uint32_t>::createBuffer(devBuffer, -1);

  auto defaultStream = GPU::Stream::getDefault();

  if (inPlace) {
    pyr->start(devGPUBuf, GPU::Buffer<uint32_t>(), defaultStream);
    pyr->compute(defaultStream);
  } else {
    pyr->start(GPU::Buffer<uint32_t>(), GPU::Buffer<uint32_t>(), defaultStream);
    pyr->compute(devGPUBuf, defaultStream);
  }
  ENSURE(defaultStream.synchronize());
  pyr->collapse(true, defaultStream);
  ENSURE(defaultStream.synchronize());
  uint32_t* actualOut = new uint32_t[w * h];
  ENSURE_CUDA(cudaMemcpy(actualOut, pyr->getLevel(0).data().get(), w * h * 4, cudaMemcpyDeviceToHost));
  ENSURE_CUDA(cudaFree(devBuffer));

  ENSURE_RGBA8888_ARRAY_EQ(in, actualOut, w, h);
  delete[] actualOut;
  delete pyr;
}

// Same, except that the pyramid is shifted 1 << numLevels pixels left in pyramid space then back to the right in image
// space. If wrapping works, then it should be a noop.
void testRoundTripWrapping(const uint32_t* in, unsigned w, unsigned h, unsigned numLevels, bool inPlace) {
  uint32_t* devBuffer;
  ENSURE_CUDA(cudaMalloc((void**)&devBuffer, w * h * 4));
  ENSURE_CUDA(cudaMemcpy(devBuffer, in, w * h * 4, cudaMemcpyHostToDevice));

  Core::LaplacianPyramid<uint32_t>* pyr =
      Core::LaplacianPyramid<uint32_t>::create("test", w, h, numLevels,
                                               inPlace ? Core::LaplacianPyramid<uint32_t>::ExternalFirstLevel
                                                       : Core::LaplacianPyramid<uint32_t>::InternalFirstLevel,
                                               Core::LaplacianPyramid<uint32_t>::Multiple, 5 /*Gaussian filter radius*/,
                                               2 /*Passes*/, true)
          .release();

  const unsigned shift = 2 << numLevels;

  auto devGPUBuf = GPU::DeviceBuffer<uint32_t>::createBuffer(devBuffer, -1);

  auto defaultStream = GPU::Stream::getDefault();

  if (inPlace) {
    pyr->start(devGPUBuf, GPU::Buffer<uint32_t>(), defaultStream);
    pyr->compute(defaultStream);
  } else {
    pyr->start(GPU::Buffer<uint32_t>(), GPU::Buffer<uint32_t>(), defaultStream);
    pyr->compute(devGPUBuf, defaultStream);
  }
  ENSURE(defaultStream.synchronize());

  // shift the pyramid left
  unsigned tShift = shift;
  for (int l = 0; l < pyr->numLevels(); ++l) {
    Core::LaplacianPyramid<uint32_t>::LevelSpec<uint32_t>& level = pyr->getLevel(l);
    std::cout << "level: " << level.width() << " x " << level.height() << " shift is " << tShift << std::endl;
    shiftDevLeft(level.data().get(), level.width(), level.height(), tShift);
    tShift /= 2;
  }
  Core::LaplacianPyramid<uint32_t>::LevelSpec<uint32_t>& level = pyr->getLevel(pyr->numLevels());
  std::cout << "level: " << level.width() << " x " << level.height() << " shift is " << tShift << std::endl;
  shiftDevLeft(level.data().get(), level.width(), level.height(), tShift);

  // collapse
  pyr->collapse(true, defaultStream);
  defaultStream.synchronize();

  uint32_t* actualOut = new uint32_t[w * h];
  ENSURE_CUDA(cudaMemcpy(actualOut, pyr->getLevel(0).data().get(), w * h * 4, cudaMemcpyDeviceToHost));
  ENSURE_CUDA(cudaFree(devBuffer));

  std::cout << "collapsed, shift is " << shift << std::endl;
  shiftHostRight(actualOut, w, h, shift);

  ENSURE_RGBA8888_ARRAY_EQ(in, actualOut, w, h);
  delete[] actualOut;
  delete pyr;
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // Fully solid
  {
    unsigned w = 536;  // must be a multiple of 1 << levels
    unsigned h = 512;
    uint32_t* in = new uint32_t[w * h];

    uint32_t v = 7;
    for (unsigned i = 0; i < w * h; ++i) {
      in[i] = VideoStitch::Image::RGBA::pack(v, v, v, 0xff);
      v = (v * 13) % 255;
    }

    VideoStitch::Testing::testRoundTrip(in, w, h, 3, false);
    VideoStitch::Testing::testRoundTrip(in, w, h, 3, true);
    VideoStitch::Testing::testRoundTripWrapping(in, w, h, 3, false);
    VideoStitch::Testing::testRoundTripWrapping(in, w, h, 3, true);
    delete[] in;
  }

  // Alpha varies solid
  {
    unsigned w = 536;  // must be a multiple of 1 << levels
    unsigned h = 512;
    uint32_t* in = new uint32_t[w * h];

    uint32_t v = 7;
    for (unsigned i = 0; i < w * h; ++i) {
      // 1 / 16 of the pixels are transparent
      int32_t alpha = ((i & 0x0f) == 0x0f) ? 0x0 : 0xff;
      in[i] = VideoStitch::Image::RGBA::pack(v, v, v, alpha);
      v = (v * 13) % 255;
    }

    VideoStitch::Testing::testRoundTrip(in, w, h, 3, false);
    VideoStitch::Testing::testRoundTrip(in, w, h, 3, true);
    VideoStitch::Testing::testRoundTripWrapping(in, w, h, 3, false);
    VideoStitch::Testing::testRoundTripWrapping(in, w, h, 3, true);
    delete[] in;
  }
  cudaDeviceReset();
  return 0;
}
