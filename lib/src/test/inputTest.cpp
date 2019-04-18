// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Basic input unpacking tests.

#include "gpu/testing.hpp"

#include "gpu/allocator.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "image/unpack.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/allocator.hpp"

namespace VideoStitch {
namespace Testing {

void testRgb888Rgba888(unsigned width, unsigned height) {
  // Generate data
  unsigned char* input = new unsigned char[3 * width * height];
  for (unsigned i = 0; i < width * height; ++i) {
    input[3 * i + 0] = (unsigned char)((33 * i) % 123);
    input[3 * i + 1] = (unsigned char)((31 * i) % 83);
    input[3 * i + 2] = (unsigned char)((42 * i) % 57);
  }

  // alloc
  auto surf = Core::OffscreenAllocator::createSourceSurface(width, height, "testRgb888Rgba888");
  ENSURE(surf.ok(), "no device memory");
  auto src = GPU::Buffer<unsigned char>::allocate(3 * width * height, "InputTest");
  ENSURE(src.status(), "no device memory");

  auto potUniqStream = GPU::UniqueStream::create();
  ENSURE(potUniqStream.status());
  auto stream = potUniqStream.ref().borrow();

  // transfer and convert
  ENSURE(GPU::memcpyAsync(src.value(), input, stream), "transfer error");
  ENSURE(Image::convertRGBToRGBA(*surf->pimpl->surface, src.value(), width, height, stream));

  // Check result of conversion
  uint32_t* outputRgba = new uint32_t[width * height];
  ENSURE(GPU::memcpyAsync(outputRgba, *surf->pimpl->surface, stream), "transfer error");
  ENSURE(stream.synchronize());
  for (unsigned i = 0; i < width * height; ++i) {
    ENSURE_EQ((int)input[3 * i + 0], (int)((unsigned char*)outputRgba)[4 * i + 0]);
    ENSURE_EQ((int)input[3 * i + 1], (int)((unsigned char*)outputRgba)[4 * i + 1]);
    ENSURE_EQ((int)input[3 * i + 2], (int)((unsigned char*)outputRgba)[4 * i + 2]);
    ENSURE_EQ(255, (int)((unsigned char*)outputRgba)[4 * i + 3]);
  }
  delete[] input;
  delete[] outputRgba;
  ENSURE(src.value().release());
}

void testPYuv420RgbaMinimal(unsigned width, unsigned height, bool display) {
  ENSURE(!(width & 1), "width must be even");
  ENSURE(!(height & 1), "height must be even");

  unsigned size = width * height + (width * height) / 4 + (width * height) / 4;

  // Generate data
  unsigned char* input = new unsigned char[size];
  for (unsigned i = 0; i < width * height; ++i) {
    input[i] = 128;
  }
  for (unsigned i = 0; i < (width * height) / 4; ++i) {
    input[width * height + i] = 123;
  }
  for (unsigned i = 0; i < (width * height) / 4; ++i) {
    input[width * height + (width * height) / 4 + i] = 156;
  }

  // alloc
  auto surf = Core::OffscreenAllocator::createSourceSurface(width, height, "testPYuv420RgbaMinimal");
  ENSURE(surf.ok(), "no device memory");
  auto src = GPU::Buffer<unsigned char>::allocate(size, "InputTest");
  ENSURE(src.status(), "no device memory");

  auto potUniqStream = GPU::UniqueStream::create();
  ENSURE(potUniqStream.status());
  auto stream = potUniqStream.ref().borrow();

  // transfer and convert
  ENSURE(GPU::memcpyAsync(src.value(), input, size, stream), "transfer error");
  ENSURE(Image::convertYV12ToRGBA(*surf->pimpl->surface, src.value(), width, height, stream));

  // Check result of conversion (should be uniform)
  uint32_t* outputRgba = new uint32_t[width * height];
  ENSURE(GPU::memcpyAsync(outputRgba, *surf->pimpl->surface, stream), "transfer error");
  ENSURE(stream.synchronize());

  if (display) {
    for (unsigned i = 0; i < height; ++i) {
      for (unsigned j = 0; j < width; ++j) {
        printf("%i %i %i %i\t\t", ((unsigned char*)outputRgba)[width * i + j],
               ((unsigned char*)outputRgba)[width * i + j + 1], ((unsigned char*)outputRgba)[width * i + j + 2],
               ((unsigned char*)outputRgba)[width * i + j + 3]);
      }
      printf("\n");
      std::cout << std::endl;
    }
  }
  for (unsigned i = 0; i < width * height; ++i) {
    ENSURE_EQ((int)((unsigned char*)outputRgba)[0], (int)((unsigned char*)outputRgba)[4 * i + 0]);
    ENSURE_EQ((int)((unsigned char*)outputRgba)[1], (int)((unsigned char*)outputRgba)[4 * i + 1]);
    ENSURE_EQ((int)((unsigned char*)outputRgba)[2], (int)((unsigned char*)outputRgba)[4 * i + 2]);
    ENSURE_EQ(255, (int)((unsigned char*)outputRgba)[4 * i + 3]);
  }
  delete[] input;
  delete[] outputRgba;
  ENSURE(src.value().release());
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testRgb888Rgba888(312, 531);
  VideoStitch::Testing::testPYuv420RgbaMinimal(16, 16, true);
  VideoStitch::Testing::testPYuv420RgbaMinimal(312, 532, false);
  return 0;
}
