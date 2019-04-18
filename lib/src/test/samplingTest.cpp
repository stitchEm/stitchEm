// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "common/testing.hpp"
#include "common/util.hpp"
#include "common/shiftBuffers.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/gpu_device.hpp"
#include <gpu/stream.hpp>
#include <gpu/image/sampling.hpp>

#include <stdint.h>

namespace VideoStitch {
namespace Testing {
void testSubsample22(const unsigned char* in, const unsigned char* expOut, unsigned inWidth, unsigned inHeight) {
  DeviceBuffer<unsigned char> devInBuffer(inWidth, inHeight);
  devInBuffer.fillData(in);
  const unsigned outWidth = (inWidth + 1) / 2;
  const unsigned outHeight = (inHeight + 1) / 2;
  DeviceBuffer<unsigned char> devOutBuffer(outWidth, outHeight);
  Image::subsample22(devOutBuffer.gpuBuf(), devInBuffer.gpuBufConst(), inWidth, inHeight, GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  std::vector<unsigned char> actualOut;
  devOutBuffer.readback(actualOut);

  for (unsigned y = 0; y < outHeight; ++y) {
    for (unsigned x = 0; x < outWidth; ++x) {
      ENSURE_EQ(expOut[y * outWidth + x], actualOut[y * outWidth + x]);
    }
  }
}
/*
void testSubsample22Mask(const uint32_t *in, const uint32_t* inMask, const uint32_t *expOut, const uint32_t* expOutMask,
unsigned inWidth, unsigned inHeight) { DeviceBuffer<uint32_t> devInBuffer(inWidth, inHeight); devInBuffer.fillData(in);

  DeviceBuffer<uint32_t> devInMaskBuffer(inWidth, inHeight);
  devInMaskBuffer.fillData(inMask);

  const unsigned outWidth = (inWidth + 1) / 2;
  const unsigned outHeight = (inHeight + 1) / 2;
  DeviceBuffer<uint32_t> devOutBuffer(outWidth, outHeight);
  DeviceBuffer<uint32_t> devOutMaskBuffer(outWidth, outHeight);

  Image::subsample22Mask<uint32_t>(devOutBuffer.gpuBuf(), devOutMaskBuffer.gpuBuf(), devInBuffer.gpuBufConst(),
devInMaskBuffer.gpuBufConst(), inWidth, inHeight, 16, GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  std::vector<uint32_t> actualOut;
  devOutBuffer.readback(actualOut);
  std::vector<uint32_t> actualOutMask;
  devOutMaskBuffer.readback(actualOutMask);

  for (unsigned y = 0; y < outHeight; ++y) {
    for (unsigned x = 0; x < outWidth; ++x) {
      ENSURE_EQ(expOut[y * outWidth + x], actualOut[y * outWidth + x]);
      ENSURE_EQ(expOutMask[y * outWidth + x], actualOutMask[y * outWidth + x]);
    }
  }
}

void testSubsample22Nearest(const uint32_t* in, const uint32_t* expOut, unsigned inWidth, unsigned inHeight) {
  DeviceBuffer<uint32_t> devInBuffer(inWidth, inHeight);
  devInBuffer.fillData(in);
  const unsigned outWidth = (inWidth + 1) / 2;
  const unsigned outHeight = (inHeight + 1) / 2;
  DeviceBuffer<uint32_t> devOutBuffer(outWidth, outHeight);
  Image::subsample22Nearest(devOutBuffer.gpuBuf(), devInBuffer.gpuBufConst(), inWidth, inHeight, 16,
GPU::Stream::getDefault()); GPU::Stream::getDefault().synchronize(); std::vector<uint32_t> actualOut;
  devOutBuffer.readback(actualOut);

  for (unsigned y = 0; y < outHeight; ++y) {
    for (unsigned x = 0; x < outWidth; ++x) {
      ENSURE_EQ(expOut[y * outWidth + x], actualOut[y * outWidth + x]);
    }
  }
}

void testSubsampleMask22(const unsigned char* in, const unsigned char* expOut, unsigned inWidth, unsigned inHeight) {
  DeviceBuffer<unsigned char> devInBuffer(inWidth, inHeight);
  devInBuffer.fillData(in);
  const unsigned outWidth = (inWidth + 1) / 2;
  const unsigned outHeight = (inHeight + 1) / 2;
  DeviceBuffer<unsigned char> devOutBuffer(outWidth, outHeight);
  Image::subsampleMask22(devOutBuffer.gpuBuf(), devInBuffer.gpuBuf(), inWidth, inHeight, 16, GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  std::vector<unsigned char> actualOut;
  devOutBuffer.readback(actualOut);

  for (unsigned y = 0; y < outHeight; ++y) {
    for (unsigned x = 0; x < outWidth; ++x) {
      ENSURE_EQ((int)expOut[y * outWidth + x], (int)actualOut[y * outWidth + x]);
    }
  }
}
*/
void testSubsample22RGBA(const uint32_t* in, const uint32_t* expOut, unsigned inWidth, unsigned inHeight) {
  DeviceBuffer<uint32_t> devInBuffer(inWidth, inHeight);
  devInBuffer.fillData(in);
  const unsigned outWidth = (inWidth + 1) / 2;
  const unsigned outHeight = (inHeight + 1) / 2;
  DeviceBuffer<uint32_t> devOutBuffer(outWidth, outHeight);
  Image::subsample22RGBA(devOutBuffer.gpuBuf(), devInBuffer.gpuBuf(), inWidth, inHeight, GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  std::vector<uint32_t> actualOut;
  devOutBuffer.readback(actualOut);

  ENSURE_RGBA8888_ARRAY_EQ(expOut, actualOut.data(), outWidth, outHeight);
}

void testUpsample22(const uint32_t* in, const uint32_t* expOut, unsigned outWidth, unsigned outHeight) {
  const unsigned inWidth = (outWidth + 1) / 2;
  const unsigned inHeight = (outHeight + 1) / 2;
  DeviceBuffer<uint32_t> devInBuffer(inWidth, inHeight);
  devInBuffer.fillData(in);
  DeviceBuffer<uint32_t> devOutBuffer(outWidth, outHeight);
  Image::upsample22(devOutBuffer.gpuBuf(), devInBuffer.gpuBufConst(), outWidth, outHeight, false,
                    GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  std::vector<uint32_t> actualOut;
  devOutBuffer.readback(actualOut);

  for (unsigned y = 0; y < outHeight; ++y) {
    for (unsigned x = 0; x < outWidth; ++x) {
      ENSURE_EQ(expOut[y * outWidth + x], actualOut[y * outWidth + x]);
    }
  }
}

void testUpsample22RGBA210(const uint32_t* inRGBA, const uint32_t* expOutRGBA, unsigned outWidth, unsigned outHeight) {
  const unsigned inWidth = (outWidth + 1) / 2;
  const unsigned inHeight = (outHeight + 1) / 2;
  DeviceBuffer<uint32_t> devInBuffer(inWidth, inHeight);
  devInBuffer.fillData(inRGBA);
  DeviceBuffer<uint32_t> devOutBuffer(outWidth, outHeight);
  Image::upsample22RGBA210(devOutBuffer.gpuBuf(), devInBuffer.gpuBuf(), outWidth, outHeight, false,
                           GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  std::vector<uint32_t> actualOut;
  devOutBuffer.readback(actualOut);

  ENSURE_RGBA210_ARRAY_EQ(expOutRGBA, actualOut.data(), outWidth, outHeight);
}

/**
 * Upsampling then subsampling is idempotent.
 * upsampling the result gives back the same result.
 */
void testRoundTripsample(const unsigned char* in, unsigned upWidth, unsigned upHeight, unsigned blockSize) {
  // Convert buffer to RGBA
  std::vector<uint32_t> upRGB;
  for (unsigned i = 0; i < upWidth * upHeight; ++i) {
    upRGB.push_back(Image::RGBA::pack(in[3 * i + 0], in[3 * i + 1], in[3 * i + 2], 0xff));
  }

  unsigned downWidth = (upWidth + 1) / 2;
  unsigned downHeight = (upHeight + 1) / 2;

  DeviceBuffer<uint32_t> devUpBuffer(upWidth, upHeight);
  DeviceBuffer<uint32_t> devUpBuffer2(upWidth, upHeight);
  devUpBuffer.fill(upRGB);
  DeviceBuffer<uint32_t> devDownBuffer(downWidth, downHeight);
  DeviceBuffer<uint32_t> devDownBuffer2(downWidth, downHeight);
  Image::subsample22RGBA(devDownBuffer.gpuBuf(), devUpBuffer.gpuBufConst(), upWidth, upHeight,
                         GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();

  // Round trips
  Image::upsample22RGBA(devUpBuffer.gpuBuf(), devDownBuffer.gpuBuf(), upWidth, upHeight, false,
                        GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  Image::subsample22RGBA(devDownBuffer2.gpuBuf(), devUpBuffer.gpuBufConst(), upWidth, upHeight,
                         GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();
  Image::upsample22RGBA(devUpBuffer2.gpuBuf(), devDownBuffer2.gpuBuf(), upWidth, upHeight, false,
                        GPU::Stream::getDefault());
  GPU::Stream::getDefault().synchronize();

  std::vector<uint32_t> hostUpBuffer;
  devUpBuffer.readback(hostUpBuffer);
  std::vector<uint32_t> hostUpBuffer2;
  devUpBuffer2.readback(hostUpBuffer2);
  std::vector<uint32_t> hostDownBuffer;
  devDownBuffer.readback(hostDownBuffer);
  std::vector<uint32_t> hostDownBuffer2;
  devDownBuffer2.readback(hostDownBuffer2);

  for (unsigned y = 0; y < downHeight; ++y) {
    for (unsigned x = 0; x < downWidth; ++x) {
      ENSURE_EQ(hostDownBuffer[y * downWidth + x], hostDownBuffer2[y * downWidth + x]);
    }
  }

  for (unsigned y = 0; y < upHeight; ++y) {
    for (unsigned x = 0; x < upWidth; ++x) {
      ENSURE_EQ(hostUpBuffer[y * upWidth + x], hostUpBuffer2[y * upWidth + x]);
    }
  }
}

/**
 * Make sure that wrapping upsampling works:
 * shifting then upsampling should be the same as subsampling then shifting.
 */
void testWrappingUpsample22RGBA210(unsigned upWidth, unsigned upHeight, unsigned blockSize, unsigned shift) {
  unsigned downWidth = (upWidth + 1) / 2;
  unsigned downHeight = (upHeight + 1) / 2;
  // Convert buffer to RGBA
  std::vector<uint32_t> downRGBA;
  int v = 7;
  for (unsigned i = 0; i < downWidth * downHeight; ++i) {
    int32_t r = v;
    v = (v * 13) % 255;
    int32_t g = v;
    v = (v * 13) % 255;
    int32_t b = v;
    v = (v * 13) % 255;
    downRGBA.push_back(Image::RGB210::pack(r, g, b, 0xff));
  }

  DeviceBuffer<uint32_t> devDownBuffer(downWidth, downHeight);
  devDownBuffer.fill(downRGBA);
  DeviceBuffer<uint32_t> devUpBuffer(upWidth, upHeight);

  // upsample unshifted
  Image::upsample22RGBA210(devUpBuffer.gpuBuf(), devDownBuffer.gpuBuf(), upWidth, upHeight, true,
                           GPU::Stream::getDefault());
  cudaStreamSynchronize(0);

  std::vector<uint32_t> hostUpBuffer;
  devUpBuffer.readback(hostUpBuffer);

  // shift the input
  shiftDevLeft(devDownBuffer.gpuBuf().get(), downWidth, downHeight, shift);

  // upsample shifted
  Image::upsample22RGBA210(devUpBuffer.gpuBuf(), devDownBuffer.gpuBuf(), upWidth, upHeight, true,
                           GPU::Stream::getDefault());
  cudaStreamSynchronize(0);
  std::vector<uint32_t> hostShiftedUpBuffer;
  devUpBuffer.readback(hostShiftedUpBuffer);

  // shift the output twice as much as the input
  shiftHostLeft(hostUpBuffer.data(), upWidth, upHeight, 2 * shift);

  ENSURE_RGBA210_ARRAY_EQ(hostUpBuffer.data(), hostShiftedUpBuffer.data(), upWidth, upHeight);
}

void testUpsampleBench(const int64_t width, const int64_t height) {
  PackedDeviceBuffer inputBuffer(width, height);
  inputBuffer.fill(5, 10, 15);
  PackedDeviceBuffer outputBuffer(2 * width, 2 * height);
  auto potStream = GPU::Stream::create();
  {
    cudaDeviceSynchronize();
    Util::SimpleProfiler p("Upsample bench", false, Logger::get(Logger::Info));
    for (int i = 0; i < 300; ++i) {
      Image::upsample22RGBA210(outputBuffer.gpuBuf(), inputBuffer.gpuBuf(), 2 * width, 2 * height, false,
                               potStream.value());
    }
    cudaDeviceSynchronize();
  }
  potStream.value().destroy();
}

void testSubsampleBench(const int64_t width, const int64_t height) {
  PackedDeviceBuffer inputBuffer(width, height);
  inputBuffer.fill(5, 10, 15);
  PackedDeviceBuffer outputBuffer((width + 1) / 2, (height + 1) / 2);
  auto potStream = GPU::Stream::create();
  {
    cudaDeviceSynchronize();
    Util::SimpleProfiler p("Subsample bench", false, Logger::get(Logger::Info));
    for (int i = 0; i < 1000; ++i) {
      Image::subsample22RGBA(outputBuffer.gpuBuf(), inputBuffer.gpuBuf(), width, height, potStream.value());
    }
    cudaDeviceSynchronize();
  }
  potStream.value().destroy();
}
}  // namespace Testing
}  // namespace VideoStitch

#define PACKSAME(v, a) VideoStitch::Image::RGBA::pack((v), (v), (v), (a))
#define PACKSAME210(v, a) VideoStitch::Image::RGB210::pack((v), (v), (v), (a))
#define NA 0

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // SubSample
  {
    unsigned char in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char expOut[] = {
        3,
        5,
        11,
        13,
    };
    VideoStitch::Testing::testSubsample22(in, expOut, 4, 4);
  }
  {
    unsigned char in[] = {1, 2, 3, 4, 91, 5, 6, 7, 8, 92, 9, 10, 11, 12, 93, 13, 14, 15, 16, 94};
    unsigned char expOut[] = {3, 5, 91, 11, 13, 93};
    VideoStitch::Testing::testSubsample22(in, expOut, 5, 4);
  }
  {
    unsigned char in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    unsigned char expOut[] = {3, 5, 11, 13, 17, 19};
    VideoStitch::Testing::testSubsample22(in, expOut, 4, 5);
  }
  {
    unsigned char in[] = {1, 2, 3, 4, 91, 5, 6, 7, 8, 92, 9, 10, 11, 12, 93, 13, 14, 15, 16, 94, 17, 18, 19, 20, 95};
    unsigned char expOut[] = {3, 5, 91, 11, 13, 93, 17, 19, 95};
    VideoStitch::Testing::testSubsample22(in, expOut, 5, 5);
  }

  /*
    //SubSample22Mask
    {
      uint32_t in[] = {
        1,  2,  3,  4, 91,
        5,  6,  7,  8, 92,
        9, 10, 11, 12, 93,
        13, 14, 15, 16, 94,
        17, 18, 19, 20, 95
      };
      uint32_t inMask[] = {
        1, 1, 1, 0, 0,
        1, 1, 1, 2, 2,
        1, 1, 1, 2, 2,
        3, 3, 3, 2, 2,
        3, 3, 3, 2, 2
      };
      uint32_t expOut[] = {
        3,  3, 91,
        9, 11, 93,
        17, 19, 95
      };
      uint32_t expOutMask[] = {
        1, 1, 0,
        1, 1, 2,
        3, 3, 2
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsample22Mask(in, inMask, expOut, expOutMask, 5, 5);
    }

    //SubSampleNearest
    {
      uint32_t in[] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 16
      };
      uint32_t expOut[] = {
        1,  3,
        9, 11,
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsample22Nearest(in, expOut, 4, 4);
    }
    {
      uint32_t in[] = {
        1,  2,  3,  4, 91,
        5,  6,  7,  8, 92,
        9, 10, 11, 12, 93,
        13, 14, 15, 16, 94
      };
      uint32_t expOut[] = {
        1,  3, 91,
        9, 11, 93
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsample22Nearest(in, expOut, 5, 4);
    }
    {
      uint32_t in[] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20
      };
      uint32_t expOut[] = {
        1,  3,
        9, 11,
        17, 19
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsample22Nearest(in, expOut, 4, 5);
    }
    {
      uint32_t in[] = {
        1,  2,  3,  4, 91,
        5,  6,  7,  8, 92,
        9, 10, 11, 12, 93,
        13, 14, 15, 16, 94,
        17, 18, 19, 20, 95
      };
      uint32_t expOut[] = {
        1,  3, 91,
        9, 11, 93,
        17, 19, 95
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsample22Nearest(in, expOut, 5, 5);
    }

    //SubSample mask
    {
      unsigned char in[] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
      };
      unsigned char expOut[] = {
        0, 0,
        0, 0,
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 4, 4);
    }
    {
      unsigned char in[] = {
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
      };
      unsigned char expOut[] = {
        1, 0,
        0, 0,
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 4, 4);
    }
    {
      unsigned char in[] = {
        0, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
      };
      unsigned char expOut[] = {
        0, 1,
        0, 0,
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 4, 4);
    }
    {
      unsigned char in[] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 0, 0, 0
      };
      unsigned char expOut[] = {
        0, 0,
        1, 0,
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 4, 4);
    }
    {
      unsigned char in[] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1
      };
      unsigned char expOut[] = {
        0, 0,
        0, 1,
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 4, 4);
    }
    {
      unsigned char in[] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  1,
        0,  0,  0,  0,  0,
      };
      unsigned char expOut[] = {
        0, 0, 0,
        0, 0, 1
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 5, 4);
    }
    {
      unsigned char in[] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  1,  0,
        0,  0,  0,  0,  0,
      };
      unsigned char expOut[] = {
        0, 0, 0,
        0, 1, 0
      };
      // XXX TODO FIXME VideoStitch::Testing::testSubsampleMask22(in, expOut, 5, 4);
    }
  */

  // Subsample, with alpha
  uint32_t in[] = {PACKSAME(1, 0),    PACKSAME(2, 0),    PACKSAME(3, 0),    PACKSAME(4, 255),  PACKSAME(91, 0),
                   PACKSAME(5, 0),    PACKSAME(6, 0),    PACKSAME(7, 0),    PACKSAME(8, 0),    PACKSAME(92, 255),
                   PACKSAME(9, 0),    PACKSAME(10, 255), PACKSAME(11, 255), PACKSAME(12, 255), PACKSAME(93, 0),
                   PACKSAME(13, 255), PACKSAME(14, 0),   PACKSAME(15, 0),   PACKSAME(16, 255), PACKSAME(94, 0),
                   PACKSAME(17, 255), PACKSAME(18, 255), PACKSAME(19, 0),   PACKSAME(20, 0),   PACKSAME(95, 255)};
  uint32_t expOut[] = {PACKSAME(NA, 0), PACKSAME(4, 255),  PACKSAME(92, 255), PACKSAME(11, 255), PACKSAME(13, 255),
                       PACKSAME(NA, 0), PACKSAME(17, 255), PACKSAME(NA, 0),   PACKSAME(95, 255)};
  VideoStitch::Testing::testSubsample22RGBA(in, expOut, 5, 5);

  // UpSample
  {
    uint32_t expOut[] = {0, 4, 12, 16, 12, 16, 24, 28, 36, 40, 48, 52, 48, 52, 60, 64};
    uint32_t in[] = {
        0,
        16,
        48,
        64,
    };
    VideoStitch::Testing::testUpsample22(in, expOut, 4, 4);
  }
  {
    uint32_t in[] = {PACKSAME210(0, 1), PACKSAME210(16, 1), PACKSAME210(48, 1), PACKSAME210(64, 1)};
    uint32_t expOut[] = {PACKSAME210(0, 1),  PACKSAME210(4, 1),  PACKSAME210(12, 1), PACKSAME210(16, 1),
                         PACKSAME210(12, 1), PACKSAME210(16, 1), PACKSAME210(24, 1), PACKSAME210(28, 1),
                         PACKSAME210(36, 1), PACKSAME210(40, 1), PACKSAME210(48, 1), PACKSAME210(52, 1),
                         PACKSAME210(48, 1), PACKSAME210(52, 1), PACKSAME210(60, 1), PACKSAME210(64, 1)};
    VideoStitch::Testing::testUpsample22RGBA210(in, expOut, 4, 4);
  }
  {
    uint32_t in[] = {
        PACKSAME210(0, 1),  PACKSAME210(16, 1), PACKSAME210(32, 1),
        PACKSAME210(48, 1), PACKSAME210(64, 1), PACKSAME210(80, 1),
    };
    uint32_t expOut[] = {PACKSAME210(0, 1),  PACKSAME210(4, 1),  PACKSAME210(12, 1), PACKSAME210(20, 1),
                         PACKSAME210(28, 1), PACKSAME210(32, 1), PACKSAME210(12, 1), PACKSAME210(16, 1),
                         PACKSAME210(24, 1), PACKSAME210(32, 1), PACKSAME210(40, 1), PACKSAME210(44, 1),
                         PACKSAME210(36, 1), PACKSAME210(40, 1), PACKSAME210(48, 1), PACKSAME210(56, 1),
                         PACKSAME210(64, 1), PACKSAME210(68, 1), PACKSAME210(48, 1), PACKSAME210(52, 1),
                         PACKSAME210(60, 1), PACKSAME210(68, 1), PACKSAME210(76, 1), PACKSAME210(80, 1)};
    VideoStitch::Testing::testUpsample22RGBA210(in, expOut, 6, 4);
  }
  {
    uint32_t in[] = {PACKSAME210(0, 1),  PACKSAME210(48, 1), PACKSAME210(16, 1),
                     PACKSAME210(64, 1), PACKSAME210(32, 1), PACKSAME210(80, 1)};
    uint32_t expOut[] = {PACKSAME210(0, 1),  PACKSAME210(12, 1), PACKSAME210(36, 1), PACKSAME210(48, 1),
                         PACKSAME210(4, 1),  PACKSAME210(16, 1), PACKSAME210(40, 1), PACKSAME210(52, 1),
                         PACKSAME210(12, 1), PACKSAME210(24, 1), PACKSAME210(48, 1), PACKSAME210(60, 1),
                         PACKSAME210(20, 1), PACKSAME210(32, 1), PACKSAME210(56, 1), PACKSAME210(68, 1),
                         PACKSAME210(28, 1), PACKSAME210(40, 1), PACKSAME210(64, 1), PACKSAME210(76, 1),
                         PACKSAME210(32, 1), PACKSAME210(44, 1), PACKSAME210(68, 1), PACKSAME210(80, 1)};
    VideoStitch::Testing::testUpsample22RGBA210(in, expOut, 4, 6);
  }
  {
    uint32_t in[] = {
        PACKSAME210(0, 1),  PACKSAME210(16, 1), PACKSAME210(32, 1),
        PACKSAME210(48, 1), PACKSAME210(64, 1), PACKSAME210(80, 1),
    };
    uint32_t expOut[] = {
        PACKSAME210(0, 1),  PACKSAME210(4, 1),  PACKSAME210(12, 1), PACKSAME210(20, 1), PACKSAME210(28, 1),
        PACKSAME210(12, 1), PACKSAME210(16, 1), PACKSAME210(24, 1), PACKSAME210(32, 1), PACKSAME210(40, 1),
        PACKSAME210(36, 1), PACKSAME210(40, 1), PACKSAME210(48, 1), PACKSAME210(56, 1), PACKSAME210(64, 1),
        PACKSAME210(48, 1), PACKSAME210(52, 1), PACKSAME210(60, 1), PACKSAME210(68, 1), PACKSAME210(76, 1),
    };
    VideoStitch::Testing::testUpsample22RGBA210(in, expOut, 5, 4);
  }
  // upsample, with alpha
  {
    uint32_t in[] = {
        PACKSAME210(0, 0),  PACKSAME210(16, 0), PACKSAME210(32, 0),
        PACKSAME210(48, 1), PACKSAME210(64, 1), PACKSAME210(80, 1),
    };
    uint32_t expOut[] = {PACKSAME210(NA, 0), PACKSAME210(NA, 0), PACKSAME210(NA, 0), PACKSAME210(NA, 0),
                         PACKSAME210(NA, 0), PACKSAME210(48, 1), PACKSAME210(52, 1), PACKSAME210(60, 1),
                         PACKSAME210(68, 1), PACKSAME210(76, 1), PACKSAME210(48, 1), PACKSAME210(52, 1),
                         PACKSAME210(60, 1), PACKSAME210(68, 1), PACKSAME210(76, 1)};
    VideoStitch::Testing::testUpsample22RGBA210(in, expOut, 5, 3);
  }

  /*{
    unsigned w = 351;
    unsigned h = 432;
    unsigned char* data = new unsigned char[3 * w * h];
    int v = 7;
    for (unsigned i = 0; i < 3 * w * h; ++i) {
      data[i] = (unsigned char)v;
      v = (v * 13) % 255;
    }
    VideoStitch::Testing::testRoundTripsample(data, w, h, 16);
    delete[] data;
  }*/

  VideoStitch::Testing::testWrappingUpsample22RGBA210(2 * 16, 2 * 16, 16, 1);
  VideoStitch::Testing::testWrappingUpsample22RGBA210(2 * 16 + 2, 2 * 16, 16, 1);
  VideoStitch::Testing::testWrappingUpsample22RGBA210(2 * 16, 16 + 1, 2 * 16, 1);
  VideoStitch::Testing::testWrappingUpsample22RGBA210(2 * 16, 2 * 16 + 2, 16, 1);
  VideoStitch::Testing::testWrappingUpsample22RGBA210(2 * 16 + 2, 2 * 16 + 2, 16, 1);

  if (argc > 1 && !strcmp(argv[1], "bench_down")) {
    std::cout << "Benching subsampling, weird dims..." << std::endl;
    VideoStitch::Testing::testSubsampleBench(4215, 2111);
    std::cout << "Benching subsampling, power of two dims..." << std::endl;
    VideoStitch::Testing::testSubsampleBench(4096, 2048);
  }

  if (argc > 1 && !strcmp(argv[1], "bench_up")) {
    std::cout << "Benching upsampling, weird dims..." << std::endl;
    VideoStitch::Testing::testUpsampleBench(4215, 2111);
    std::cout << "Benching upsampling, power of two dims..." << std::endl;
    VideoStitch::Testing::testUpsampleBench(4096, 2048);
  }

  cudaDeviceReset();
  return 0;
}
