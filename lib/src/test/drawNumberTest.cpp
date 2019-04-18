// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "gpu/util.hpp"

#include "gpu/allocator.hpp"
#include <gpu/render/render.hpp>
#include <gpu/render/numberDrafter.hpp>
#include "libvideostitch/gpu_device.hpp"

namespace VideoStitch {
namespace Testing {

void testFillBuffer() {
  auto uniqStream = GPU::UniqueStream::create();
  ENSURE(uniqStream.status());
  auto stream = uniqStream.ref().borrow();

  PackedDeviceBuffer reference(257, 509);
  reference.fill(0x44, 0x55, 0x00);

  PackedDeviceBuffer buffer(257, 509);
  buffer.fill(0x0, 0x0, 0x0);

  // buffer (blockingly) filled by the testing framework, should not match reference
  buffer.ENSURE_NEQ(reference);

  // buffer async filled by library function
  ENSURE(Render::fillBuffer(buffer.gpuBuf(), Image::RGBA::pack(0x44, 0x55, 0x00, 0xff), 257, 509, stream));
  ENSURE(stream.synchronize());

  // should match reference now
  buffer.ENSURE_EQ(reference);
}

void testDrawNumbers() {
  auto uniqStream = GPU::UniqueStream::create();
  ENSURE(uniqStream.status());
  auto stream = uniqStream.ref().borrow();

  size_t outputWidth = 1901;
  size_t outputHeight = 313;

  DeviceBuffer<uint32_t> reference(outputWidth, outputHeight);
  reference.readPngFromFile("data/render/numbers.png");

  DeviceBuffer<uint32_t> buffer(outputWidth, outputHeight);
  buffer.fill(0);

  auto drawColor = Image::RGBA::pack(0x37, 0xaa, 0xef, 0xff);
  Render::NumberDrafter drafter(outputWidth / 12);
  Render::fillBuffer(buffer.gpuBuf(), Image::RGBA::pack(0x00, 0x00, 0x0, 1), outputWidth, outputHeight, stream);
  GPU::Buffer<uint32_t> b = buffer.gpuBuf();
  for (int i = 0; i < 10; i++) {
    drafter.draw(i, b, outputWidth, outputHeight, outputWidth / 10 * i + 15, 10, drawColor, stream);
  }
  stream.synchronize();

  // create new reference image:
  // DeviceBuffer<uint32_t> res(outputWidth, outputHeight, (uint32_t*)buffer.data());
  // res.dumpToPng8888("data/render/numbers.png");

  buffer.ENSURE_BUF_EQ(reference);
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  VideoStitch::Testing::testFillBuffer();
  VideoStitch::Testing::testDrawNumbers();
  return 0;
}
