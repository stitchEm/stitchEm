// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "gpu/core1/voronoi.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"

#include "util/pnm.hpp"

#include "libvideostitch/gpu_device.hpp"

namespace VideoStitch {
namespace Testing {
#define fromIdMask 0x10
#define toIdMask 0x0f

unsigned char* run(int width, int height, const uint32_t* input, bool wrap) {
  auto src = GPU::uniqueBuffer<uint32_t>(width * height, "VoronoiTest");
  auto work = GPU::uniqueBuffer<uint32_t>(width * height, "VoronoiTest");
  auto dst = GPU::uniqueBuffer<unsigned char>(width * height, "VoronoiTest");

  ENSURE(src.status());
  ENSURE(work.status());
  ENSURE(dst.status());

  auto potUniqStream = GPU::UniqueStream::create();
  ENSURE(potUniqStream.status());
  auto stream = potUniqStream.ref().borrow();

  // transfer and compute
  ENSURE(GPU::memcpyAsync(src.borrow(), input, stream));
  Core::voronoiCompute(dst.borrow(), src.borrow(), work.borrow(), width, height, fromIdMask, toIdMask, wrap,
                       std::min(16, width), stream);
  unsigned char* output = new unsigned char[width * height];
  ENSURE(GPU::memcpyAsync(output, dst.borrow().as_const(), stream));

  stream.synchronize();

  return output;
}

void testVoronoiSmallCorners() {
  // Generate data
  uint32_t* input = new uint32_t[4 * 4];
  for (int i = 1; i < 4 * 4 - 1; ++i) {
    input[i] = 0;
  }
  input[0] = fromIdMask;
  input[4 * 4 - 1] = toIdMask;

  unsigned char* output = run(4, 4, input, false);

  // Check results.
  /*for (int i = 0 ; i < 4 * 4; ++i) {
    std::cout << i << " -> " << (int)output[i] << std::endl;
  }*/
  // w w w ?
  // w w ? b
  // w ? b b
  // ? b b b
  ENSURE_EQ(255, (int)output[0]);
  ENSURE_EQ(255, (int)output[1]);
  ENSURE_EQ(255, (int)output[2]);
  ENSURE_EQ(255, (int)output[4]);
  ENSURE_EQ(255, (int)output[5]);
  ENSURE_EQ(255, (int)output[8]);

  ENSURE_EQ(0, (int)output[7]);
  ENSURE_EQ(0, (int)output[10]);
  ENSURE_EQ(0, (int)output[11]);
  ENSURE_EQ(0, (int)output[13]);
  ENSURE_EQ(0, (int)output[14]);
  ENSURE_EQ(0, (int)output[15]);
  delete[] output;

  output = run(4, 4, input, true);
  /*for (int i = 0 ; i < 4 * 4; ++i) {
    std::cout << i << " -> " << (int)output[i] << std::endl;
  }*/
  // w w w w
  // w w ? w
  // b ? b b
  // b b b b
  ENSURE_EQ(255, (int)output[0]);
  ENSURE_EQ(255, (int)output[1]);
  ENSURE_EQ(255, (int)output[2]);
  ENSURE_EQ(255, (int)output[3]);
  ENSURE_EQ(255, (int)output[4]);
  ENSURE_EQ(255, (int)output[5]);
  ENSURE_EQ(255, (int)output[7]);

  ENSURE_EQ(0, (int)output[8]);
  ENSURE_EQ(0, (int)output[10]);
  ENSURE_EQ(0, (int)output[11]);
  ENSURE_EQ(0, (int)output[12]);
  ENSURE_EQ(0, (int)output[13]);
  ENSURE_EQ(0, (int)output[14]);
  ENSURE_EQ(0, (int)output[15]);
  delete[] input;
  delete[] output;
}

void testVoronoi(int width, int height) {
  int x1 = width / 2;
  int x2 = width / 3;
  int y1 = height / 2;
  int y2 = height / 3;
  int d1 = (width / 7) * (width / 7);
  int d2 = (width / 10) * (width / 10);

  // Generate data
  uint32_t* input = new uint32_t[width * height];
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      input[y * width + x] = (!!((x - x1) * (x - x1) + (y - y1) * (y - y1) < d1)) * fromIdMask |
                             (!!((x - x2) * (x - x2) + (y - y2) * (y - y2) < d2)) * toIdMask;
    }
  }
  // Shift everything a third of the width horizontally
  uint32_t* shiftedInput = new uint32_t[width * height];
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      shiftedInput[y * width + x] = input[y * width + ((x + width / 3) % width)];
    }
  }

  // Write
  {
    std::ofstream* ofs = Util::PpmWriter::openPam("nonshifted.pam", width, height, &std::cerr);
    ofs->write((const char*)input, width * height * 4);
    delete ofs;
  }
  {
    std::ofstream* ofs = Util::PpmWriter::openPam("shifted.pam", width, height, &std::cerr);
    ofs->write((const char*)shiftedInput, width * height * 4);
    delete ofs;
  }

  unsigned char* output = run(width, height, input, true);
  unsigned char* shiftedOutput = run(width, height, shiftedInput, true);

  // Write
  {
    std::ofstream* ofs = Util::PpmWriter::openPgm("nonshifted-mask.pam", width, height, &std::cerr);
    ofs->write((const char*)output, width * height);
    delete ofs;
  }
  {
    std::ofstream* ofs = Util::PpmWriter::openPgm("shifted-mask.pam", width, height, &std::cerr);
    ofs->write((const char*)shiftedOutput, width * height);
    delete ofs;
  }

  // Check results. The voronoi diagram should just be shifted.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      ENSURE_EQ((int)shiftedOutput[y * width + x], (int)output[y * width + ((x + width / 3) % width)]);
    }
  }

  delete[] output;
  delete[] shiftedOutput;
  delete[] input;
  delete[] shiftedInput;
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testVoronoiSmallCorners();
  VideoStitch::Testing::testVoronoi(512, 256);
  return 0;
}
