// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "common/testing.hpp"
#include "common/util.hpp"

#include <gpu/buffer.hpp>
#include <gpu/memcpy.hpp>
#include <gpu/image/blur.hpp>
#include "libvideostitch/gpu_device.hpp"

#include <util/pnm.hpp>

#include <stdint.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <sstream>

namespace VideoStitch {
namespace Testing {

/**
 * Golden brute-force implementation of 1D NoWrap box blur.
 */
std::vector<unsigned char> golden1DBoxBlurNoWrap(const std::vector<unsigned char>& data, int radius) {
  ENSURE(!data.empty());
  std::vector<unsigned char> extended;
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

  std::vector<unsigned char> result(data.size());
  for (int i = 0; i < (int)data.size(); ++i) {
    int v = 0;
    for (int j = 0; j < 2 * radius + 1; ++j) {
      v += extended[i + j];
    }
    ENSURE(v / (2 * radius + 1) < 255);
    result[i] = (unsigned char)(v / (2 * radius + 1));
  }
  return result;
}

/**
 * Golden brute-force implementation of 1D Wrap box blur.
 */
std::vector<unsigned char> golden1DBoxBlurWrap(const std::vector<unsigned char>& data, int radius) {
  ENSURE(!data.empty());
  std::vector<unsigned char> extended;
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

  std::vector<unsigned char> result(data.size());
  for (int i = 0; i < (int)data.size(); ++i) {
    int acc(0);
    for (int j = 0; j < 2 * radius + 1; ++j) {
      acc += extended[i + j];
    }
    result[i] = static_cast<unsigned char>(acc / (2 * radius + 1));
  }
  return result;
}

void testBoxBlurNoWrap(int size, int radius) {
  std::vector<unsigned char> data;
  for (int i = 0; i < size; ++i) {
    data.push_back((unsigned char)(((i + 1) * 457) % 255));
  }
  const std::vector<unsigned char> golden = golden1DBoxBlurNoWrap(data, radius);

  DeviceBuffer<unsigned char> devSrc(1, data.size());
  devSrc.fill(data);
  DeviceBuffer<unsigned char> devDst(1, data.size());
  devDst.fill(0);

  Image::boxBlur1DNoWrap(devDst.gpuBuf(), devSrc.gpuBufConst(), 1, data.size(), radius, 16, GPU::Stream::getDefault());
  std::vector<unsigned char> actual;
  devDst.readback(actual);

  ENSURE_ARRAY_EQ(golden.data(), actual.data(), (unsigned)data.size());
}

void testBoxBlurWrap(unsigned size, unsigned radius) {
  if ((std::size_t)(2 * radius) >= size) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(size / 2 - 1);
  }
  std::vector<unsigned char> data;
  for (unsigned i = 0; i < size; ++i) {
    data.push_back((unsigned char)(((i + 1) * 457) % 255));
  }
  const std::vector<unsigned char> golden = golden1DBoxBlurWrap(data, radius);

  DeviceBuffer<unsigned char> devSrc(1, data.size());
  devSrc.fill(data);
  DeviceBuffer<unsigned char> devDst(1, data.size());
  devDst.fill(0);

  Image::boxBlur1DWrap(devDst.gpuBuf(), devSrc.gpuBufConst(), 1, data.size(), radius, 16, GPU::Stream::getDefault());
  std::vector<unsigned char> actual;
  devDst.readback(actual);

  ENSURE_ARRAY_EQ(golden.data(), actual.data(), (unsigned)data.size());
}

PotentialValue<GPU::Buffer<unsigned char>> loadFile(const char* filename, int64_t& width, int64_t& height) {
  std::vector<unsigned char> tmp;
  if (!VideoStitch::Util::PnmReader::read(filename, width, height, tmp, &std::cerr)) {
    std::stringstream msg;
    msg << "Image '" << filename << "': failed to setup reader.";
    return Status{Origin::Input, ErrType::SetupFailure, msg.str()};
  }
  std::vector<unsigned char> buffer;
  buffer.reserve((size_t)(width * height));
  for (size_t i = 0; i < (size_t)(width * height); ++i) {
    buffer[i] = tmp[(size_t)(3 * i)];
  }
  auto devBuffer = GPU::Buffer<unsigned char>::allocate((size_t)(width * height), "BlurTest");
  ENSURE(devBuffer.ok());
  ENSURE(GPU::memcpyBlocking(devBuffer.value(), &buffer.front()).ok());
  return devBuffer;
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // No Wrap
  VideoStitch::Testing::testBoxBlurNoWrap(1531, 5);
  VideoStitch::Testing::testBoxBlurNoWrap(1043, 10);  // test blur1DKernelNoWrap
  VideoStitch::Testing::testBoxBlurNoWrap(4, 2);
  VideoStitch::Testing::testBoxBlurNoWrap(4, 5);

  // Wrap
  VideoStitch::Testing::testBoxBlurWrap(1531, 5);
  VideoStitch::Testing::testBoxBlurWrap(1043, 10);  // test blur1DKernelWrap
  VideoStitch::Testing::testBoxBlurWrap(4, 2);
  VideoStitch::Testing::testBoxBlurWrap(4, 5);

  VideoStitch::Testing::testBoxBlurWrap(15, 7);  // blur1DKernelWrap

  if (argc < 4) {
    // std::cerr << "usage: " << argv[0] << " src.pgm radius passes" << std::endl;;
    return 0;
  }

  int64_t width, height;
  auto devBuffer = VideoStitch::Testing::loadFile(argv[1], width, height);
  unsigned radius = atoi(argv[2]);
  unsigned passes = atoi(argv[3]);
  std::cerr << "radius: " << radius << " passes: " << passes << std::endl;

  if (devBuffer.ok()) {
    auto devWork = VideoStitch::GPU::Buffer<unsigned char>::allocate((size_t)(width * height), "BlurTest");
    VideoStitch::Testing::ENSURE(devWork.status());
    VideoStitch::Image::gaussianBlur2D(devBuffer.value(), devWork.value(), width, height, radius, passes, false, 256,
                                       VideoStitch::GPU::Stream::getDefault());
    unsigned char* out = new unsigned char[(size_t)(width * height)];
    VideoStitch::GPU::Stream::getDefault().synchronize();
    VideoStitch::Testing::ENSURE(VideoStitch::GPU::memcpyBlocking(out, devBuffer.value()));
    std::ofstream* ofs = VideoStitch::Util::PpmWriter::openPpm("blured.pgm", width, height, &std::cerr);
    ofs->write((const char*)out, width * height);
    delete ofs;
    delete[] out;
    VideoStitch::Testing::ENSURE(devWork.value().release());
    VideoStitch::Testing::ENSURE(devBuffer.value().release());
  }

  return 0;
}
