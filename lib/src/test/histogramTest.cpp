// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Basic histogram tests.

#include "common/testing.hpp"

#include <gpu/buffer.hpp>
#include <backend/cuda/deviceBuffer.hpp>
#include <cuda/util.hpp>
#include <image/histogram.hpp>
#include <image/histogramView.hpp>
#include "libvideostitch/input.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"

#define SIZE (10 * 1024)

namespace VideoStitch {
namespace Testing {

namespace {
/**
 * A dummy reader that simply returns a SIZExSIZE image with random grayscale pixels.
 */
class RandomReader : public Input::VideoReader {
 public:
  RandomReader()
      : Input::Reader(0),
        Input::VideoReader(SIZE, SIZE, 0, PixelFormat::RGBA, Host, {-1, 0}, 0, 10, true /* procedural */, NULL) {}

  Input::ReadStatus readFrame(mtime_t&, unsigned char* data) {
    for (int y = 0; y < (int)getHeight(); ++y) {
      for (int x = 0; x < (int)getWidth(); ++x) {
        data[(y * getWidth() + x) * 3] = (unsigned char)rand() % 256;
        data[(y * getWidth() + x) * 3 + 1] = (unsigned char)rand() % 256;
        data[(y * getWidth() + x) * 3 + 2] = (unsigned char)rand() % 256;
      }
    }
    return Input::ReadStatus::OK();
  }

  Status seekFrame(frameid_t) {
    ENSURE(false, "never called");
    return Status::OK();
  }

  Status unpackMonoDevBuffer(const GPU::Buffer<unsigned char>& dst, const GPU::Buffer<const unsigned char>& src,
                             GPU::Stream&) const {
    unsigned char* tmp;
    cudaMallocHost((void**)&tmp, (size_t)(getWidth() * getHeight()));
    for (int y = 0; y < (int)getHeight(); ++y) {
      for (int x = 0; x < (int)getWidth(); ++x) {
        tmp[y * getWidth() + x] = src.get().raw()[(y * getWidth() + x) * 3];
      }
    }
    cudaMemcpy(dst.get(), tmp, (size_t)(getWidth() * getHeight()), cudaMemcpyHostToDevice);
    cudaFreeHost(tmp);
    return Status::OK();
  }
};
}  // namespace

// class LumaHistogramTest {
// public:
//  LumaHistogramTest()
//    : reader(new RandomReader()) {
//  }
//
//  ~LumaHistogramTest() {
//    delete reader;
//  }
//
//  void testHistogram() {
//    unsigned char *hostFrame;
//    cudaMallocHost((void**)&hostFrame, SIZE * SIZE * 4);
//    reader->readFrame(hostFrame);
//    unsigned char *devFrame;
//    cudaMalloc((void**)&devFrame, SIZE * SIZE);
//    GPU::Stream s;
//    reader->unpackMonoDevBuffer(GPU::Buffer<unsigned char>(devFrame, -1), GPU::Buffer<const unsigned char>(hostFrame,
//    -1), s); cudaFreeHost(hostFrame);
//
//    // Gpu histogram
//    unsigned *gpuHisto;
//    unsigned *devHist;
//    {
//      cudaMallocHost((void**)&gpuHisto ,256 * sizeof(uint32_t));
//      cudaMalloc((void**)&devHist, 256 * sizeof(uint32_t));
//
//      Util::SimpleProfiler prof("GPU luma histogram", false, Logger::get(Logger::Info));
//      Image::lumaHistogram(devFrame, SIZE, SIZE, devHist);
//    }
//    cudaMemcpy(gpuHisto, devHist, sizeof(uint32_t) * 256, cudaMemcpyDeviceToHost);
//    cudaFree(devHist);
//
//    // Cpu histogram
//    unsigned char *hostGrayscaleFrame;
//    cudaMallocHost((void**)&hostGrayscaleFrame, SIZE * SIZE);
//    unsigned cpuHisto[256];
//    for (unsigned i = 0; i < 256; ++i) {
//      cpuHisto[i] = 0;
//    }
//    {
//      cudaMemcpy(hostGrayscaleFrame, devFrame, SIZE * SIZE, cudaMemcpyDeviceToHost);
//      cudaFree(devFrame);
//
//      Util::SimpleProfiler prof("CPU luma histogram", false, Logger::get(Logger::Info));
//      for (unsigned i = 0; i < SIZE; ++i) {
//        for (unsigned j = 0; j < SIZE; ++j) {
//          cpuHisto[hostGrayscaleFrame[j + SIZE * i]]++;
//        }
//      }
//    }
//    cudaFreeHost(hostGrayscaleFrame);
//
//    // Check
//    for (unsigned i = 0; i < 256; ++i) {
//      ENSURE_EQ(gpuHisto[i], cpuHisto[i]);
//    }
//
//    cudaFreeHost(gpuHisto);
//  }
//
// private:
//  LumaHistogramTest(const LumaHistogramTest&);
//
//  Input::Reader* reader;
//};

void testBoxConvolve() {
  double src[256];
  double dst[256];
  for (int i = 0; i < 256; ++i) {
    src[i] = 0.0;
  }
  Image::boxConvolve(dst, src, 3);
  for (int i = 0; i < 256; ++i) {
    ENSURE_APPROX_EQ(0.0, dst[i], 0.0001);
  }
  src[0] = 1.0;
  src[128] = 2.0;
  src[255] = 3.0;
  Image::boxConvolve(dst, src, 3);

  ENSURE_APPROX_EQ(4 / 7.0, dst[0], 0.0001);
  ENSURE_APPROX_EQ(3 / 7.0, dst[1], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[2], 0.0001);
  ENSURE_APPROX_EQ(1 / 7.0, dst[3], 0.0001);
  ENSURE_APPROX_EQ(0.0, dst[4], 0.0001);

  ENSURE_APPROX_EQ(0.0, dst[124], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[125], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[126], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[127], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[128], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[129], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[130], 0.0001);
  ENSURE_APPROX_EQ(2 / 7.0, dst[131], 0.0001);
  ENSURE_APPROX_EQ(0.0, dst[132], 0.0001);

  ENSURE_APPROX_EQ(0.0, dst[251], 0.0001);
  ENSURE_APPROX_EQ(3 / 7.0, dst[252], 0.0001);
  ENSURE_APPROX_EQ(6 / 7.0, dst[253], 0.0001);
  ENSURE_APPROX_EQ(9 / 7.0, dst[254], 0.0001);
  ENSURE_APPROX_EQ(12 / 7.0, dst[255], 0.0001);
}

void testDistances() {
  uint32_t histo1[256];
  for (int i = 0; i < 256; ++i) {
    histo1[i] = 0;
  }
  uint32_t histo2[256];
  for (int i = 0; i < 256; ++i) {
    histo2[i] = 0;
  }
  histo1[10] = 2;
  histo2[10] = 3;
  ENSURE_APPROX_EQ(0.0, Image::CpuHistogramView(histo1).sqrDistanceL2(Image::CpuHistogramView(histo2)), 0.0001);
  ENSURE_APPROX_EQ(0.0, Image::CpuHistogramView(histo1).sqrDistanceChi2(Image::CpuHistogramView(histo2)), 0.0001);
  ENSURE_APPROX_EQ(0.0, Image::CpuHistogramView(histo1).sqrDistanceQF(Image::CpuHistogramView(histo2), 2), 0.0001);

  histo1[9] = 4;
  histo2[9] = 3;
  // ((2 / 6) - (3 / 6))^2 + ((4 / 6) - (3 / 6))^2 == 1/36 + 1/36
  ENSURE_APPROX_EQ(1 / 18.0, Image::CpuHistogramView(histo1).sqrDistanceL2(Image::CpuHistogramView(histo2)), 0.0001);
  // ((2 / 6) - (3 / 6))^2 / ((2 / 6) + (3 / 6)) + ((4 / 6) - (3 / 6))^2 /((4 / 6) + (3 / 6)) == 6/(36 * 5) + 6/(36 * 7)
  ENSURE_APPROX_EQ(1.0 / 30.0 + 1.0 / 42.0,
                   Image::CpuHistogramView(histo1).sqrDistanceChi2(Image::CpuHistogramView(histo2)), 0.0001);
  ENSURE_APPROX_EQ(0.000444444, Image::CpuHistogramView(histo1).sqrDistanceQF(Image::CpuHistogramView(histo2), 2),
                   0.0001);

  //   histo1[10] = 6;
  //   histo2[10] = 0;
  //   histo1[9] = 0;
  //   histo2[9] = 6;
  //   // 1.0^2 + 1.0^2 == 2.0
  //   ENSURE_APPROX_EQ(2.0, Image::CpuHistogramView(histo1).sqrDistanceL2(Image::CpuHistogramView(histo2)), 0.0001);
  //   //1.0^2 / 1.0 + 1.0^2 /1.0 == 2.0
  //   ENSURE_APPROX_EQ(2.0, Image::CpuHistogramView(histo1).sqrDistanceChi2(Image::CpuHistogramView(histo2)), 0.0001);
  //   ENSURE_APPROX_EQ(0.000444444, Image::CpuHistogramView(histo1).sqrDistanceQF(Image::CpuHistogramView(histo2), 2),
  //   0.0001);
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  /*cudaSetDevice(0);
  VideoStitch::Testing::LumaHistogramTest test;
  test.testHistogram();*/

  VideoStitch::Testing::testBoxConvolve();
  VideoStitch::Testing::testDistances();
  return 0;
}
