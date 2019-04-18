// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "gpu/util.hpp"
#include "common/shiftBuffers.hpp"

#include <gpu/image/blur.hpp>
#include <gpu/buffer.hpp>
#include "libvideostitch/context.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/gpu_device.hpp"

#include "image/transpose.hpp"

namespace VideoStitch {
namespace Testing {

#define PACKSAME(v, a) VideoStitch::Image::RGB210::pack((v), (v), (v), (a))
#define NA 0

/**
 * Golden brute-force implementation of 1D No Wrap box blur.
 */
std::vector<uint32_t> golden1DBoxBlurNoWrapRGBA210(const std::vector<uint32_t>& data, int radius) {
  ENSURE(!data.empty());
  std::vector<uint32_t> extended;
  // Extend edges.
  for (int i = 0; i < radius; ++i) {
    extended.push_back(data.front());
  }
  for (int i = 0; i < (int)data.size(); ++i) {
    extended.push_back(data[i]);
  }
  for (int i = 0; i < radius; ++i) {
    extended.push_back(data.back());
  }

  std::vector<uint32_t> result(data.size());
  for (int i = 0; i < (int)data.size(); ++i) {
    int r = 0;
    int g = 0;
    int b = 0;
    int divider = 0;
    for (int j = 0; j < 2 * radius + 1; ++j) {
      if (Image::RGB210::a(extended[i + j]) != 0) {
        ;
        r += Image::RGB210::r(extended[i + j]);
        g += Image::RGB210::g(extended[i + j]);
        b += Image::RGB210::b(extended[i + j]);
        divider++;
      }
    }
    if (divider != 0) {
      result[i] = Image::RGB210::pack(r / divider, g / divider, b / divider, Image::RGB210::a(data[i]));
    }
  }
  return result;
}

/**
 * Golden brute-force implementation of 1D Wrap box blur.
 */
std::vector<uint32_t> golden1DBoxBlurWrapRGBA210(const std::vector<uint32_t>& data, int radius) {
  ENSURE(!data.empty());
  std::vector<uint32_t> extended;
  // Extend edges.
  for (int i = 0; i < radius; ++i) {
    extended.push_back(data[data.size() - (radius - i)]);
  }
  for (int i = 0; i < (int)data.size(); ++i) {
    extended.push_back(data[i]);
  }
  for (int i = 0; i < radius; ++i) {
    extended.push_back(data[i]);
  }

  std::vector<uint32_t> result(data.size());
  for (int i = 0; i < (int)data.size(); ++i) {
    int r = 0;
    int g = 0;
    int b = 0;
    int divider = 0;
    for (int j = 0; j < 2 * radius + 1; ++j) {
      if (Image::RGB210::a(extended[i + j]) != 0) {
        r += Image::RGB210::r(extended[i + j]);
        g += Image::RGB210::g(extended[i + j]);
        b += Image::RGB210::b(extended[i + j]);
        divider++;
      }
    }
    if (divider != 0) {
      result[i] = Image::RGB210::pack(r / divider, g / divider, b / divider, Image::RGB210::a(data[i]));
    }
  }
  return result;
}

void testBlur1D(const uint32_t* in, uint32_t* const expectedOut, unsigned size) {
  auto devInBuffer = GPU::Buffer<uint32_t>::allocate(size, "TestBlur1D");
  ENSURE(devInBuffer.ok());
  ENSURE(GPU::memcpyBlocking(devInBuffer.value(), in));

  auto devOutBuffer = GPU::Buffer<uint32_t>::allocate(size, "TestBlur1D");
  ENSURE(devOutBuffer.ok());

  Image::boxBlurColumnsNoWrapRGBA210(devOutBuffer.value(), devInBuffer.value(), 1, size, 1, GPU::Stream::getDefault());
  ENSURE(devInBuffer.value().release());

  uint32_t* actualOut = new uint32_t[size];
  ENSURE(GPU::memcpyBlocking(actualOut, devOutBuffer.value()));
  ENSURE(devOutBuffer.value().release());

  ENSURE_RGBA210_ARRAY_EQ(expectedOut, actualOut, size, 1);
  delete[] actualOut;
}

// If wrapping, shifted buffers should have shifted blurred versions.
void testBlur1DWrapping(uint32_t* in, unsigned size) {
  auto devInBuffer = GPU::Buffer<uint32_t>::allocate(size, "TestBlur1DWrapping");
  ENSURE(devInBuffer.ok());

  auto devOutBuffer = GPU::Buffer<uint32_t>::allocate(size, "TestBlur1DWrapping");
  ENSURE(devOutBuffer.ok());

  uint32_t* actualOut = new uint32_t[size];
  uint32_t* actualShiftedOut = new uint32_t[size];

  ENSURE(GPU::memcpyBlocking(devInBuffer.value(), in));
  Image::boxBlurColumnsWrapRGBA210(devOutBuffer.value(), devInBuffer.value(), 1, size, 1, GPU::Stream::getDefault());
  ENSURE(GPU::memcpyBlocking(actualOut, devOutBuffer.value()));

  const unsigned shift = 2;
  shiftHostRight(in, size, 1, shift);
  ENSURE(GPU::memcpyBlocking(devInBuffer.value(), in));
  Image::boxBlurColumnsWrapRGBA210(devOutBuffer.value(), devInBuffer.value(), 1, size, 1, GPU::Stream::getDefault());
  ENSURE(GPU::memcpyBlocking(actualShiftedOut, devOutBuffer.value()));

  ENSURE(devInBuffer.value().release());
  ENSURE(devOutBuffer.value().release());

  shiftHostRight(actualOut, size, 1, shift);
  ENSURE_RGBA210_ARRAY_EQ(actualShiftedOut, actualOut, size, 1);

  delete[] actualOut;
  delete[] actualShiftedOut;
}

void testBoxBlurColumnsNoWrap(int size, int radius) {
  std::vector<uint32_t> data;
  for (int i = 0; i < size; ++i) {
    data.push_back(PACKSAME(rand(), i % 3));
  }
  const std::vector<uint32_t> golden = golden1DBoxBlurNoWrapRGBA210(data, radius);
  DeviceBuffer<uint32_t> devSrc(1, data.size());
  devSrc.fill(data);
  DeviceBuffer<uint32_t> devDst(1, data.size());
  devDst.fill(0);

  Image::boxBlurColumnsNoWrapRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), 1, data.size(), radius,
                                     GPU::Stream::getDefault());
  std::vector<uint32_t> actual;
  devDst.readback(actual);
  ENSURE_RGBA210_ARRAY_EQ(golden.data(), actual.data(), (unsigned)data.size(), 1);
}

void testBoxBlurColumnsWrap(unsigned size, unsigned radius) {
  if ((std::size_t)(2 * radius) >= size) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(size / 2 - 1);
  }
  std::vector<uint32_t> data;
  for (unsigned i = 0; i < size; ++i) {
    data.push_back(PACKSAME(rand(), i % 3));
  }
  const std::vector<uint32_t> golden = golden1DBoxBlurWrapRGBA210(data, radius);
  DeviceBuffer<uint32_t> devSrc(1, data.size());
  devSrc.fill(data);
  DeviceBuffer<uint32_t> devDst(1, data.size());
  devDst.fill(0);

  Image::boxBlurColumnsWrapRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), 1, data.size(), radius,
                                   GPU::Stream::getDefault());

  std::vector<uint32_t> actual;
  devDst.readback(actual);
  ENSURE_RGBA210_ARRAY_EQ(golden.data(), actual.data(), (unsigned)data.size(), 1);
}

void testBoxBlurRowsWrap(unsigned size, unsigned radius) {
  if (radius > (ROWS_BLOCKDIM_X * ROWS_HALO_STEPS)) {
    radius = ROWS_BLOCKDIM_X * ROWS_HALO_STEPS;
  }

  if ((std::size_t)(2 * radius) >= size) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(size / 2 - 1);
  }

  std::vector<uint32_t> data;
  for (unsigned i = 0; i < size; ++i) {
    data.push_back(PACKSAME(rand(), i % 3));
  }
  const std::vector<uint32_t> golden = golden1DBoxBlurWrapRGBA210(data, radius);
  DeviceBuffer<uint32_t> devSrc(1, data.size());
  devSrc.fill(data);
  DeviceBuffer<uint32_t> devDst(1, data.size());
  devDst.fill(0);

  Image::boxBlurRowsRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), data.size(), 1, radius, GPU::Stream::getDefault(),
                            true);

  std::vector<uint32_t> actual;
  devDst.readback(actual);
  ENSURE_RGBA210_ARRAY_EQ(golden.data(), actual.data(), (unsigned)data.size(), 1);
}

void testBoxBlurRowsNoWrap(unsigned size, unsigned radius) {
  if (radius > (ROWS_BLOCKDIM_X * ROWS_HALO_STEPS)) {
    radius = ROWS_BLOCKDIM_X * ROWS_HALO_STEPS;
  }
  std::vector<uint32_t> data;
  for (unsigned i = 0; i < size; ++i) {
    data.push_back(PACKSAME(rand(), i % 3));
  }
  const std::vector<uint32_t> golden = golden1DBoxBlurNoWrapRGBA210(data, radius);
  DeviceBuffer<uint32_t> devSrc(1, data.size());
  devSrc.fill(data);
  DeviceBuffer<uint32_t> devDst(1, data.size());
  devDst.fill(0);

  Image::boxBlurRowsRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), data.size(), 1, radius, GPU::Stream::getDefault(),
                            false);

  std::vector<uint32_t> actual;
  devDst.readback(actual);
  ENSURE_RGBA210_ARRAY_EQ(golden.data(), actual.data(), (unsigned)data.size(), 1);
}

void benchBlur(const int width, const int height, const unsigned radius) {
  PackedDeviceBuffer devInBuffer(width, height);
  devInBuffer.fill(5, 120, 240);

  auto stream = createTestedStream();
  {
    PackedDeviceBuffer devOutBuffer(width, height);
    PackedDeviceBuffer devWorkBuffer(width, height);
    stream.synchronize();
    Util::SimpleProfiler p("blur bench", false, Logger::get(Logger::Info));
    for (int i = 0; i < 50; ++i) {
      Image::gaussianBlur2DRGB210(devOutBuffer.gpuBuf(), devInBuffer.gpuBufConst(), devWorkBuffer.gpuBuf(), width,
                                  height, radius, 2, false, stream);
    }
    stream.synchronize();
  }
  stream.destroy();
}

void testBoxBlurRows2D(unsigned width, unsigned height, int radius, bool wrap) {
  if (radius > (ROWS_BLOCKDIM_X * ROWS_HALO_STEPS)) {
    radius = ROWS_BLOCKDIM_X * ROWS_HALO_STEPS;
  }

  if ((std::size_t)(2 * radius) >= width) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(width / 2 - 1);
  }
  std::vector<std::vector<uint32_t>> data(height);
  std::vector<uint32_t> data1D;
  std::vector<uint32_t> res;
  // will compute a golden 2D box Blur
  for (unsigned j = 0; j < height; ++j) {
    for (unsigned i = 0; i < width; ++i) {
      uint32_t value = PACKSAME(rand(), (i + j) % 3);
      data[j].push_back(value);
      data1D.push_back(value);
    }
    if (wrap) {
      std::vector<uint32_t> golden = golden1DBoxBlurWrapRGBA210(data[j], radius);
      for (unsigned k = 0; k < golden.size(); k++) {
        res.push_back(golden[k]);
      }
    } else {
      std::vector<uint32_t> golden = golden1DBoxBlurNoWrapRGBA210(data[j], radius);
      for (unsigned k = 0; k < golden.size(); k++) {
        res.push_back(golden[k]);
      }
    }
  }
  DeviceBuffer<uint32_t> devSrc(width, height);
  devSrc.fill(data1D);
  DeviceBuffer<uint32_t> devDst(width, height);
  devDst.fill(0);
  Image::boxBlurRowsRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), width, height, radius, GPU::Stream::getDefault(),
                            wrap);
  std::vector<uint32_t> actual;
  devDst.readback(actual);
  ENSURE_RGBA210_ARRAY_EQ(res.data(), actual.data(), width, height);
}

void testBoxBlurColumns2D(int width, int height, int radius, bool wrap) {
  std::vector<std::vector<uint32_t>> data(width);
  std::vector<uint32_t> data1D(width * height);
  std::vector<uint32_t> res(width * height);
  // will compute a golden 2D box Blur
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      data[j].push_back(PACKSAME(rand(), (i + j) % 3));
    }
    if (wrap) {
      std::vector<uint32_t> golden = golden1DBoxBlurWrapRGBA210(data[j], radius);
      // store result (transposed)
      for (unsigned k = 0; k < golden.size(); k++) {
        res[k * width + j] = golden[k];
      }
    } else {
      std::vector<uint32_t> golden = golden1DBoxBlurNoWrapRGBA210(data[j], radius);
      // store result (transposed)
      for (unsigned k = 0; k < golden.size(); k++) {
        res[k * width + j] = golden[k];
      }
    }
  }
  // transpose the data
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      data1D[i * width + j] = data[j][i];
    }
  }
  DeviceBuffer<uint32_t> devSrc(width, height);
  devSrc.fill(data1D);
  DeviceBuffer<uint32_t> devDst(width, height);
  devDst.fill(0);
  if (wrap) {
    Image::boxBlurColumnsWrapRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), width, height, radius,
                                     GPU::Stream::getDefault());
  } else {
    Image::boxBlurColumnsNoWrapRGBA210(devDst.gpuBuf(), devSrc.gpuBufConst(), width, height, radius,
                                       GPU::Stream::getDefault());
  }
  std::vector<uint32_t> actual;
  devDst.readback(actual);
  ENSURE_RGBA210_ARRAY_EQ(res.data(), actual.data(), width, height);
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  ENSURE(VideoStitch::GPU::setThreadLocalBackendDeviceIndex(0));
  {
    uint32_t in[] = {PACKSAME(1, 0xff),  PACKSAME(2, 0xff),  PACKSAME(4, 0xff),  PACKSAME(8, 0xff),
                     PACKSAME(16, 0xff), PACKSAME(32, 0xff), PACKSAME(64, 0xff), PACKSAME(128, 0xff)};
    uint32_t out[] = {PACKSAME(1, 0xff),  PACKSAME(2, 0xff),  PACKSAME(4, 0xff),  PACKSAME(9, 0xff),
                      PACKSAME(18, 0xff), PACKSAME(37, 0xff), PACKSAME(74, 0xff), PACKSAME(106, 0xff)};
    VideoStitch::Testing::testBlur1D(in, out, 8);
    VideoStitch::Testing::testBlur1DWrapping(in, 8);  // destroys in
  }
  {
    uint32_t in[] = {PACKSAME(1, 0x00),  PACKSAME(2, 0xff),  PACKSAME(4, 0x00),  PACKSAME(8, 0xff),
                     PACKSAME(16, 0xff), PACKSAME(32, 0xff), PACKSAME(64, 0x00), PACKSAME(128, 0x00)};
    uint32_t out[] = {PACKSAME(NA, 0x00), PACKSAME(2, 0xff),   // 2 / 1
                      PACKSAME(NA, 0x00), PACKSAME(12, 0xff),  // (8 + 16) / 2
                      PACKSAME(18, 0xff),                      // (8 + 16 + 32) / 3
                      PACKSAME(24, 0xff),                      // (16 + 32) / 2
                      PACKSAME(NA, 0x00), PACKSAME(NA, 0x00)};
    VideoStitch::Testing::testBlur1D(in, out, 8);
    VideoStitch::Testing::testBlur1DWrapping(in, 8);  // destroys in
  }

  VideoStitch::Testing::testBoxBlurColumnsNoWrap(97, 10);  // blur1DKernelNoWrap
  VideoStitch::Testing::testBoxBlurColumnsNoWrap(103, 6);  // blurColumnsKernelNoWrap

  VideoStitch::Testing::testBoxBlurColumnsWrap(88, 6);    // blurColumnsKernelWrap
  VideoStitch::Testing::testBoxBlurColumnsWrap(124, 10);  // blur1DKernelWrap

  VideoStitch::Testing::testBoxBlurRowsWrap(149, 3);
  VideoStitch::Testing::testBoxBlurRowsNoWrap(8, 2);

  VideoStitch::Testing::testBoxBlurRows2D(32, 17, 4, false);

  VideoStitch::Testing::testBoxBlurRows2D(32, 17, 18, false);

  VideoStitch::Testing::testBoxBlurRows2D(16, 27, 4, true);
  VideoStitch::Testing::testBoxBlurRows2D(16, 27, 10, true);

  VideoStitch::Testing::testBoxBlurColumns2D(94, 19, 3, true);
  VideoStitch::Testing::testBoxBlurColumns2D(78, 134, 7, false);

  VideoStitch::GPU::Context::destroy();
  if (argc > 1 && !strcmp(argv[1], "bench")) {
    VideoStitch::Testing::benchBlur(4541, 3211, 1);
  }
  return 0;
}
