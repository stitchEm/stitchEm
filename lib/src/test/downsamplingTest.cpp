// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/util.hpp"
#include "gpu/testing.hpp"

#include "gpu/2dBuffer.hpp"
#include <gpu/image/downsampler.hpp>
#include "libvideostitch/context.hpp"

#include <stdint.h>
#include <stdlib.h>

namespace VideoStitch {
namespace Testing {

void testDownsampleRGBA(const int64_t width, const int64_t height) {
  std::vector<unsigned char> input(width * height * 4);
  std::vector<unsigned char> expectedOutput(width * height);
  for (int64_t y = 0; y < height / 2; ++y) {
    for (int64_t x = 0; x < width / 2; ++x) {
      int expectedR = 0;
      int expectedG = 0;
      int expectedB = 0;

      unsigned char r = (unsigned char)(rand() % 255);
      unsigned char g = (unsigned char)(rand() % 255);
      unsigned char b = (unsigned char)(rand() % 255);
      input[4 * (2 * (y * width + x))] = r;
      input[4 * (2 * (y * width + x)) + 1] = g;
      input[4 * (2 * (y * width + x)) + 2] = b;
      input[4 * (2 * (y * width + x)) + 3] = 255;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      r = (unsigned char)(rand() % 255);
      g = (unsigned char)(rand() % 255);
      b = (unsigned char)(rand() % 255);
      input[4 * (2 * (y * width + x) + 1)] = r;
      input[4 * (2 * (y * width + x) + 1) + 1] = g;
      input[4 * (2 * (y * width + x) + 1) + 2] = b;
      input[4 * (2 * (y * width + x) + 1) + 3] = 255;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      r = (unsigned char)(rand() % 255);
      g = (unsigned char)(rand() % 255);
      b = (unsigned char)(rand() % 255);
      input[4 * (2 * (y * width + x) + width)] = r;
      input[4 * (2 * (y * width + x) + width) + 1] = g;
      input[4 * (2 * (y * width + x) + width) + 2] = b;
      input[4 * (2 * (y * width + x) + width) + 3] = 255;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      r = (unsigned char)(rand() % 255);
      g = (unsigned char)(rand() % 255);
      b = (unsigned char)(rand() % 255);
      input[4 * (2 * (y * width + x) + width + 1)] = r;
      input[4 * (2 * (y * width + x) + width + 1) + 1] = g;
      input[4 * (2 * (y * width + x) + width + 1) + 2] = b;
      input[4 * (2 * (y * width + x) + width + 1) + 3] = 255;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      expectedOutput[4 * (y * width / 2 + x)] = (unsigned char)(expectedR / 4);
      expectedOutput[4 * (y * width / 2 + x) + 1] = (unsigned char)(expectedG / 4);
      expectedOutput[4 * (y * width / 2 + x) + 2] = (unsigned char)(expectedB / 4);
      expectedOutput[4 * (y * width / 2 + x) + 3] = 255;
    }
  }
  GPU::Buffer2D in = GPU::Buffer2D::allocate(width * 4, height, "input").value();
  ENSURE(GPU::memcpyBlocking(in, input.data()), "transfer error");
  GPU::Buffer2D out = GPU::Buffer2D::allocate(width * 2, height / 2, "output").value();

  ENSURE(Image::downsample(PixelFormat::RGBA, &in, &out, GPU::Stream::getDefault()));
  ENSURE(GPU::Stream::getDefault().synchronize());
  in.release();

  std::vector<unsigned char> actual(width * height);
  ENSURE(GPU::memcpyBlocking(actual.data(), out), "transfer error");
  out.release();

  ENSURE_ARRAY_EQ((unsigned char*)expectedOutput.data(), (unsigned char*)actual.data(), expectedOutput.size());
}

void testDownsampleRGB(const int64_t width, const int64_t height) {
  std::vector<unsigned char> input(width * height * 3);
  std::vector<unsigned char> expectedOutput((width * height * 3) / (2 * 2));
  for (int64_t y = 0; y < height / 2; ++y) {
    for (int64_t x = 0; x < width / 2; ++x) {
      int expectedR = 0;
      int expectedG = 0;
      int expectedB = 0;

      unsigned char r = (unsigned char)(rand() % 255);
      unsigned char g = (unsigned char)(rand() % 255);
      unsigned char b = (unsigned char)(rand() % 255);
      input[3 * (2 * (y * width + x))] = r;
      input[3 * (2 * (y * width + x)) + 1] = g;
      input[3 * (2 * (y * width + x)) + 2] = b;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      r = (unsigned char)(rand() % 255);
      g = (unsigned char)(rand() % 255);
      b = (unsigned char)(rand() % 255);
      input[3 * (2 * (y * width + x) + 1)] = r;
      input[3 * (2 * (y * width + x) + 1) + 1] = g;
      input[3 * (2 * (y * width + x) + 1) + 2] = b;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      r = (unsigned char)(rand() % 255);
      g = (unsigned char)(rand() % 255);
      b = (unsigned char)(rand() % 255);
      input[3 * (2 * (y * width + x) + width)] = r;
      input[3 * (2 * (y * width + x) + width) + 1] = g;
      input[3 * (2 * (y * width + x) + width) + 2] = b;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      r = (unsigned char)(rand() % 255);
      g = (unsigned char)(rand() % 255);
      b = (unsigned char)(rand() % 255);
      input[3 * (2 * (y * width + x) + width + 1)] = r;
      input[3 * (2 * (y * width + x) + width + 1) + 1] = g;
      input[3 * (2 * (y * width + x) + width + 1) + 2] = b;
      expectedR += r;
      expectedG += g;
      expectedB += b;

      expectedOutput[3 * (y * width / 2 + x)] = (unsigned char)(expectedR / 4);
      expectedOutput[3 * (y * width / 2 + x) + 1] = (unsigned char)(expectedG / 4);
      expectedOutput[3 * (y * width / 2 + x) + 2] = (unsigned char)(expectedB / 4);
    }
  }
  GPU::Buffer2D in = GPU::Buffer2D::allocate(width * 3, height, "input").value();
  ENSURE(GPU::memcpyBlocking(in, input.data()), "transfer error");
  GPU::Buffer2D out = GPU::Buffer2D::allocate(width * 3 / 2, height / 2, "output").value();

  ENSURE(Image::downsample(PixelFormat::RGB, &in, &out, GPU::Stream::getDefault()));
  ENSURE(GPU::Stream::getDefault().synchronize());
  in.release();

  std::vector<unsigned char> actual((width * height * 3) / (2 * 2));
  ENSURE(GPU::memcpyBlocking(actual.data(), out), "transfer error");
  out.release();

  ENSURE_ARRAY_EQ((unsigned char*)expectedOutput.data(), (unsigned char*)actual.data(), expectedOutput.size());
}

void testDownsampleYV12(const int64_t width, const int64_t height) {
  std::vector<unsigned char> inputY(width * height);
  std::vector<unsigned char> inputU(width * height / 4);
  std::vector<unsigned char> inputV(width * height / 4);
  std::vector<unsigned char> expectedOutputY(width * height / 4);
  std::vector<unsigned char> expectedOutputU(width * height / 16);
  std::vector<unsigned char> expectedOutputV(width * height / 16);
  // Fill planes
  for (int64_t y = 0; y < height / 2; ++y) {
    for (int64_t x = 0; x < width / 2; ++x) {
      int expectedV = 0;

      unsigned char v = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x))] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x) + 1)] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x) + width)] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x) + width + 1)] = v;
      expectedV += v;

      expectedOutputY[(y * width / 2 + x)] = (unsigned char)(expectedV / 4);
    }
  }
  for (int64_t y = 0; y < height / 4; ++y) {
    for (int64_t x = 0; x < width / 4; ++x) {
      int expectedV = 0;

      unsigned char v = (unsigned char)(rand() % 255);
      inputU[(2 * (y * width / 2 + x))] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputU[(2 * (y * width / 2 + x) + 1)] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputU[(2 * (y * width / 2 + x) + width / 2)] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputU[(2 * (y * width / 2 + x) + width / 2 + 1)] = v;
      expectedV += v;

      expectedOutputU[(y * width / 4 + x)] = (unsigned char)(expectedV / 4);
    }
  }
  for (int64_t y = 0; y < height / 4; ++y) {
    for (int64_t x = 0; x < width / 4; ++x) {
      int expectedV = 0;

      unsigned char v = (unsigned char)(rand() % 255);
      inputV[(2 * (y * width / 2 + x))] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputV[(2 * (y * width / 2 + x) + 1)] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputV[(2 * (y * width / 2 + x) + width / 2)] = v;
      expectedV += v;

      v = (unsigned char)(rand() % 255);
      inputV[(2 * (y * width / 2 + x) + width / 2 + 1)] = v;
      expectedV += v;

      expectedOutputV[(y * width / 4 + x)] = (unsigned char)(expectedV / 4);
    }
  }

  GPU::Buffer2D in[3];
  in[0] = GPU::Buffer2D::allocate(width, height, "inputY").value();
  ENSURE(GPU::memcpyBlocking(in[0], inputY.data()), "transfer error");
  in[1] = GPU::Buffer2D::allocate(width / 2, height / 2, "inputU").value();
  ENSURE(GPU::memcpyBlocking(in[1], inputU.data()), "transfer error");
  in[2] = GPU::Buffer2D::allocate(width / 2, height / 2, "inputV").value();
  ENSURE(GPU::memcpyBlocking(in[2], inputV.data()), "transfer error");
  GPU::Buffer2D out[3];
  out[0] = GPU::Buffer2D::allocate(width / 2, height / 2, "outputY").value();
  out[1] = GPU::Buffer2D::allocate(width / 4, height / 4, "outputU").value();
  out[2] = GPU::Buffer2D::allocate(width / 4, height / 4, "outputV").value();

  ENSURE(Image::downsample(PixelFormat::YV12, in, out, GPU::Stream::getDefault()));
  ENSURE(GPU::Stream::getDefault().synchronize());
  in[0].release();
  in[1].release();
  in[2].release();

  std::vector<unsigned char> actualY(width * height / 4);
  std::vector<unsigned char> actualU(width * height / 16);
  std::vector<unsigned char> actualV(width * height / 16);
  ENSURE(GPU::memcpyBlocking(actualY.data(), out[0]), "transfer error");
  ENSURE(GPU::memcpyBlocking(actualU.data(), out[1]), "transfer error");
  ENSURE(GPU::memcpyBlocking(actualV.data(), out[2]), "transfer error");
  out[0].release();
  out[1].release();
  out[2].release();

  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputY.data(), (unsigned char*)actualY.data(), expectedOutputY.size());
  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputU.data(), (unsigned char*)actualU.data(), expectedOutputU.size());
  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputV.data(), (unsigned char*)actualV.data(), expectedOutputV.size());
}

void testDownsampleNV12(const int64_t width, const int64_t height) {
  std::vector<unsigned char> inputY(width * height);
  std::vector<unsigned char> inputUV(width * height / 2);
  std::vector<unsigned char> expectedOutputY(width * height / 4);
  std::vector<unsigned char> expectedOutputUV(width * height / 8);
  // Fill planes
  // Y
  for (int64_t y = 0; y < height / 2; ++y) {
    for (int64_t x = 0; x < width / 2; ++x) {
      int expectedY = 0;

      unsigned char Y = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x))] = Y;
      expectedY += Y;

      Y = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x) + 1)] = Y;
      expectedY += Y;

      Y = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x) + width)] = Y;
      expectedY += Y;

      Y = (unsigned char)(rand() % 255);
      inputY[(2 * (y * width + x) + width + 1)] = Y;
      expectedY += Y;

      expectedOutputY[(y * width / 2 + x)] = (unsigned char)(expectedY / 4);
    }
  }
  // UV
  for (int64_t y = 0; y < height / 4; ++y) {
    for (int64_t x = 0; x < width / 2; x += 2) {
      int expectedU = 0, expectedV = 0;

      unsigned char U = (unsigned char)(rand() % 255);
      unsigned char V = (unsigned char)(rand() % 255);
      inputUV[2 * y * width + 2 * x] = U;
      inputUV[2 * y * width + 2 * x + 1] = V;
      expectedU += U;
      expectedV += V;

      U = (unsigned char)(rand() % 255);
      V = (unsigned char)(rand() % 255);
      inputUV[2 * y * width + 2 * x + 2] = U;
      inputUV[2 * y * width + 2 * x + 3] = V;
      expectedU += U;
      expectedV += V;

      U = (unsigned char)(rand() % 255);
      V = (unsigned char)(rand() % 255);
      inputUV[(2 * y + 1) * width + 2 * x] = U;
      inputUV[(2 * y + 1) * width + 2 * x + 1] = V;
      expectedU += U;
      expectedV += V;

      U = (unsigned char)(rand() % 255);
      V = (unsigned char)(rand() % 255);
      inputUV[(2 * y + 1) * width + 2 * x + 2] = U;
      inputUV[(2 * y + 1) * width + 2 * x + 3] = V;
      expectedU += U;
      expectedV += V;

      expectedOutputUV[(y * width / 2 + x)] = (unsigned char)(expectedU / 4);
      expectedOutputUV[(y * width / 2 + x + 1)] = (unsigned char)(expectedV / 4);
    }
  }

  GPU::Buffer2D in[2];
  in[0] = GPU::Buffer2D::allocate(width, height, "inputY").value();
  ENSURE(GPU::memcpyBlocking(in[0], inputY.data()), "transfer error");
  in[1] = GPU::Buffer2D::allocate(width, height / 2, "inputUV").value();
  ENSURE(GPU::memcpyBlocking(in[1], inputUV.data()), "transfer error");
  GPU::Buffer2D out[2];
  out[0] = GPU::Buffer2D::allocate(width / 2, height / 2, "outputY").value();
  out[1] = GPU::Buffer2D::allocate(width / 2, height / 4, "outputUV").value();

  ENSURE(Image::downsample(PixelFormat::NV12, in, out, GPU::Stream::getDefault()));
  ENSURE(GPU::Stream::getDefault().synchronize());
  in[0].release();
  in[1].release();

  std::vector<unsigned char> actualY(width * height / 4);
  std::vector<unsigned char> actualUV(width * height / 8);
  ENSURE(GPU::memcpyBlocking(actualY.data(), out[0]), "transfer error");
  ENSURE(GPU::memcpyBlocking(actualUV.data(), out[1]), "transfer error");
  out[0].release();
  out[1].release();

  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputY.data(), (unsigned char*)actualY.data(), expectedOutputY.size());
  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputUV.data(), (unsigned char*)actualUV.data(), expectedOutputUV.size());
}

void testDownsampleYUV422(const int64_t width, const int64_t height) {
  std::vector<unsigned char> input(width * height * 2);
  std::vector<unsigned char> expectedOutput(width * height / 2);
  // each iteration writes 2 destination pixels / 8 source pixels, like 4x2
  // x and y in destination pixels
  for (int64_t y = 0; y < height / 2; ++y) {
    for (int64_t x = 0; x < width / 2; x += 2) {
      int expectedU = 0;
      int expectedY0 = 0;
      int expectedV = 0;
      int expectedY1 = 0;

      // X and Y in source pixels
      int64_t X = 2 * x;
      int64_t Y = 2 * y;

      // 2 source pixels
      unsigned char u = (unsigned char)(rand() % 255);
      unsigned char y0 = (unsigned char)(rand() % 255);
      unsigned char v = (unsigned char)(rand() % 255);
      unsigned char y1 = (unsigned char)(rand() % 255);
      input[2 * Y * width + 2 * X] = u;
      input[2 * Y * width + 2 * X + 1] = y0;
      input[2 * Y * width + 2 * X + 2] = v;
      input[2 * Y * width + 2 * X + 3] = y1;
      expectedU += u;
      expectedY0 += y0;
      expectedV += v;
      expectedY0 += y1;

      // 2 source pixels
      u = (unsigned char)(rand() % 255);
      y0 = (unsigned char)(rand() % 255);
      v = (unsigned char)(rand() % 255);
      y1 = (unsigned char)(rand() % 255);
      input[2 * Y * width + 2 * (X + 2)] = u;
      input[2 * Y * width + 2 * (X + 2) + 1] = y0;
      input[2 * Y * width + 2 * (X + 2) + 2] = v;
      input[2 * Y * width + 2 * (X + 2) + 3] = y1;
      expectedU += u;
      expectedY1 += y0;
      expectedV += v;
      expectedY1 += y1;

      // 2 source pixels
      u = (unsigned char)(rand() % 255);
      y0 = (unsigned char)(rand() % 255);
      v = (unsigned char)(rand() % 255);
      y1 = (unsigned char)(rand() % 255);
      input[2 * (Y + 1) * width + 2 * X] = u;
      input[2 * (Y + 1) * width + 2 * X + 1] = y0;
      input[2 * (Y + 1) * width + 2 * X + 2] = v;
      input[2 * (Y + 1) * width + 2 * X + 3] = y1;
      expectedU += u;
      expectedY0 += y0;
      expectedV += v;
      expectedY0 += y1;

      // 2 source pixels
      u = (unsigned char)(rand() % 255);
      y0 = (unsigned char)(rand() % 255);
      v = (unsigned char)(rand() % 255);
      y1 = (unsigned char)(rand() % 255);
      input[2 * (Y + 1) * width + 2 * (X + 2)] = u;
      input[2 * (Y + 1) * width + 2 * (X + 2) + 1] = y0;
      input[2 * (Y + 1) * width + 2 * (X + 2) + 2] = v;
      input[2 * (Y + 1) * width + 2 * (X + 2) + 3] = y1;
      expectedU += u;
      expectedY1 += y0;
      expectedV += v;
      expectedY1 += y1;

      // 2 destination pixels
      expectedOutput[y * width + 2 * x] = (unsigned char)(expectedU / 4);
      expectedOutput[y * width + 2 * x + 1] = (unsigned char)(expectedY0 / 4);
      expectedOutput[y * width + 2 * x + 2] = (unsigned char)(expectedV / 4);
      expectedOutput[y * width + 2 * x + 3] = (unsigned char)(expectedY1 / 4);
    }
  }
  GPU::Buffer2D in = GPU::Buffer2D::allocate(width * 2, height, "input").value();
  ENSURE(GPU::memcpyBlocking(in, input.data()), "transfer error");
  GPU::Buffer2D out = GPU::Buffer2D::allocate(width, height / 2, "output").value();

  ENSURE(Image::downsample(PixelFormat::UYVY, &in, &out, GPU::Stream::getDefault()));
  ENSURE(GPU::Stream::getDefault().synchronize());
  in.release();

  std::vector<unsigned char> actual(width * height / 2);
  ENSURE(GPU::memcpyBlocking(actual.data(), out), "transfer error");
  out.release();

  ENSURE_ARRAY_EQ((unsigned char*)expectedOutput.data(), (unsigned char*)actual.data(), expectedOutput.size());
}

void testDownsampleYUV422P10(const int64_t width, const int64_t height) {
  std::vector<uint16_t> inputY(width * height);
  std::vector<uint16_t> inputU(width * height / 2);
  std::vector<uint16_t> inputV(width * height / 2);
  std::vector<uint16_t> expectedOutputY(width * height / 4);
  std::vector<uint16_t> expectedOutputU(width * height / 8);
  std::vector<uint16_t> expectedOutputV(width * height / 8);

  // this test will not work with factors other than 2
  // adding this variable to keep track of difference between image downsampling
  // and chroma subsampling factors
  const int64_t downsamplingFactor = 2;

  // 4:2:0 --> half horizontal resolution, full vertical resolution
  const int64_t subsamplingFactor = 2;

  // Fill planes
  // Y
  for (int64_t y = 0; y < height / downsamplingFactor; ++y) {
    for (int64_t x = 0; x < width / downsamplingFactor; ++x) {
      int64_t expectedY = 0;

      uint16_t Y = (uint16_t)(rand() % 1023);
      inputY[(2 * (y * width + x))] = Y;
      expectedY += Y;

      Y = (uint16_t)(rand() % 1023);
      inputY[(2 * (y * width + x) + 1)] = Y;
      expectedY += Y;

      Y = (uint16_t)(rand() % 1023);
      inputY[(2 * (y * width + x) + width)] = Y;
      expectedY += Y;

      Y = (uint16_t)(rand() % 1023);
      inputY[(2 * (y * width + x) + width + 1)] = Y;
      expectedY += Y;

      expectedOutputY[(y * width / downsamplingFactor + x)] = (uint16_t)(expectedY / 4);
    }
  }
  for (int64_t y = 0; y < height / downsamplingFactor; ++y) {
    for (int64_t x = 0; x < width / downsamplingFactor / subsamplingFactor; ++x) {
      int64_t expectedV = 0;

      uint16_t v = (uint16_t)(rand() % 1023);
      inputU[(2 * (y * width / 2 + x))] = v;
      expectedV += v;

      v = (uint16_t)(rand() % 1023);
      inputU[(2 * (y * width / 2 + x) + 1)] = v;
      expectedV += v;

      v = (uint16_t)(rand() % 1023);
      inputU[(2 * (y * width / 2 + x) + width / 2)] = v;
      expectedV += v;

      v = (uint16_t)(rand() % 1023);
      inputU[(2 * (y * width / 2 + x) + width / 2 + 1)] = v;
      expectedV += v;

      expectedOutputU[(y * width / downsamplingFactor / subsamplingFactor + x)] = (uint16_t)(expectedV / 4);
    }
  }
  for (int64_t y = 0; y < height / downsamplingFactor; ++y) {
    for (int64_t x = 0; x < width / downsamplingFactor / subsamplingFactor; ++x) {
      int64_t expectedV = 0;

      uint16_t v = (uint16_t)(rand() % 1023);
      inputV[(2 * (y * width / 2 + x))] = v;
      expectedV += v;

      v = (uint16_t)(rand() % 1023);
      inputV[(2 * (y * width / 2 + x) + 1)] = v;
      expectedV += v;

      v = (uint16_t)(rand() % 1023);
      inputV[(2 * (y * width / 2 + x) + width / 2)] = v;
      expectedV += v;

      v = (uint16_t)(rand() % 1023);
      inputV[(2 * (y * width / 2 + x) + width / 2 + 1)] = v;
      expectedV += v;

      expectedOutputV[(y * width / downsamplingFactor / subsamplingFactor + x)] = (uint16_t)(expectedV / 4);
    }
  }

  GPU::Buffer2D in[3];
  in[0] = GPU::Buffer2D::allocate(width * 2, height, "inputY").value();
  ENSURE(GPU::memcpyBlocking(in[0], (unsigned char*)inputY.data()), "transfer error");
  in[1] = GPU::Buffer2D::allocate(width, height, "inputU").value();
  ENSURE(GPU::memcpyBlocking(in[1], (unsigned char*)inputU.data()), "transfer error");
  in[2] = GPU::Buffer2D::allocate(width, height, "inputV").value();
  ENSURE(GPU::memcpyBlocking(in[2], (unsigned char*)inputV.data()), "transfer error");
  GPU::Buffer2D out[3];
  out[0] = GPU::Buffer2D::allocate(width, height / 2, "outputY").value();
  out[1] = GPU::Buffer2D::allocate(width / 2, height / 2, "outputU").value();
  out[2] = GPU::Buffer2D::allocate(width / 2, height / 2, "outputV").value();

  ENSURE(Image::downsample(PixelFormat::YUV422P10, in, out, GPU::Stream::getDefault()));
  ENSURE(GPU::Stream::getDefault().synchronize());
  in[0].release();
  in[1].release();
  in[2].release();

  std::vector<uint16_t> actualY(width * height / 4);
  std::vector<uint16_t> actualU(width * height / 8);
  std::vector<uint16_t> actualV(width * height / 8);
  ENSURE(GPU::memcpyBlocking((unsigned char*)actualY.data(), out[0]), "transfer error");
  ENSURE(GPU::memcpyBlocking((unsigned char*)actualU.data(), out[1]), "transfer error");
  ENSURE(GPU::memcpyBlocking((unsigned char*)actualV.data(), out[2]), "transfer error");
  out[0].release();
  out[1].release();
  out[2].release();

  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputY.data(), (unsigned char*)actualY.data(), expectedOutputY.size());
  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputU.data(), (unsigned char*)actualU.data(), expectedOutputU.size());
  ENSURE_ARRAY_EQ((unsigned char*)expectedOutputV.data(), (unsigned char*)actualV.data(), expectedOutputV.size());
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  VideoStitch::Testing::testDownsampleRGBA(46, 14);
  VideoStitch::Testing::testDownsampleRGB(46, 14);
  VideoStitch::Testing::testDownsampleYV12(92, 28);
  VideoStitch::Testing::testDownsampleNV12(92, 28);
  VideoStitch::Testing::testDownsampleYUV422(92, 28);
  VideoStitch::Testing::testDownsampleYUV422(8, 2);
  VideoStitch::Testing::testDownsampleYUV422P10(92, 28);

  VideoStitch::Testing::ENSURE(VideoStitch::GPU::Context::destroy());
  return 0;
}
