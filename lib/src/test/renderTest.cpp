// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Basic MultiFileInput tests

#include "common/testing.hpp"
#include "common/util.hpp"
#include "libvideostitch/gpu_device.hpp"

#include <render/fillRenderer.hpp>

namespace VideoStitch {
namespace Testing {

void testFillRenderer() {
  const uint32_t color = Image::RGBA::pack(0x00, 0xff, 0x22, 0xff);
  const uint32_t bgcolor = Image::RGBA::pack(0x11, 0xff, 0x33, 0xff);

  PackedDeviceBuffer buffer(73, 117);
  const Render::FillRenderer renderer;
  // Whole area
  {
    buffer.fill(0x44, 0x55, 0x00);
    renderer.draw(buffer.ptr(), buffer.width, buffer.height, 0, 0, 73, 117, color, bgcolor, (cudaStream_t)NULL);
    PackedDeviceBuffer reference(73, 117);
    reference.readPngFromFile("data/render/fill1.png");
    buffer.ENSURE_EQ(reference);
  }
  // In the middle
  {
    buffer.fill(0x44, 0x55, 0x00);
    renderer.draw(buffer.ptr(), buffer.width, buffer.height, 15, 72, 42, 110, color, bgcolor, (cudaStream_t)NULL);
    buffer.dumpToPng("dump.png");
    PackedDeviceBuffer reference(73, 117);
    reference.readPngFromFile("data/render/fill2.png");
    buffer.ENSURE_EQ(reference);
  }
  // On the border, cropped
  {
    buffer.fill(0x44, 0x55, 0x00);
    renderer.draw(buffer.ptr(), buffer.width, buffer.height, 15, 72, 42, 220, color, bgcolor, (cudaStream_t)NULL);
    buffer.dumpToPng("dump.png");
    PackedDeviceBuffer reference(73, 117);
    reference.readPngFromFile("data/render/fill3.png");
    buffer.ENSURE_EQ(reference);
  }
  // In the corner, cropped
  {
    buffer.fill(0x44, 0x55, 0x00);
    renderer.draw(buffer.ptr(), buffer.width, buffer.height, 15, 72, 300, 220, color, bgcolor, (cudaStream_t)NULL);
    buffer.dumpToPng("dump.png");
    PackedDeviceBuffer reference(73, 117);
    reference.readPngFromFile("data/render/fill4.png");
    buffer.ENSURE_EQ(reference);
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testFillRenderer();
  return 0;
}
