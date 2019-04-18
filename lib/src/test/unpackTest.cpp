// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Basic input unpacking tests.

#include "gpu/testing.hpp"
#include "gpu/util.hpp"

#include "libvideostitch/context.hpp"
#include "gpu/allocator.hpp"
#include "gpu/2dBuffer.hpp"

#include <image/unpack.hpp>
#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"

// Uncomment this to compare with opencv. You need to have OpenCV buit with GPU support, and add opencv_gpu to your
// libs. #define WITH_OPENCV_CMP

#ifdef WITH_OPENCV_CMP
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#endif

#include <memory>

#ifdef DUMP_IMAGES
#include "util/debugUtils.hpp"
#endif

namespace {
struct NoProfiler {
  NoProfiler(const char*, bool, VideoStitch::ThreadSafeOstream&) {}
};
}  // namespace

using namespace VideoStitch::Image;

namespace VideoStitch {
namespace Testing {

template <class ProfilerT>
void testOutOfPlaceRGB(unsigned width, unsigned height) {
  // Generate data
  std::vector<unsigned char> input;
  for (unsigned i = 0; i < width * height; ++i) {
    input.push_back((unsigned char)((33 * i) % 235));
    input.push_back((unsigned char)((31 * i) % 83));
    input.push_back((unsigned char)((42 * i) % 57));
  }
  DeviceBuffer<uchar3> inputBuffer(width, height);
  inputBuffer.fillData(input);

  auto potUniqueStream = GPU::UniqueStream::create();
  ENSURE(potUniqueStream.ok());
  auto stream = potUniqueStream.ref().borrow();

  DeviceBuffer<uint32_t> rgb210Buffer(width, height);
  {
    ENSURE(stream.synchronize());
    ProfilerT p("RGB pack", false, Logger::get(Logger::Info));
    // packUnpack(PixelArray<Image::RGB210Pixel>(width, height, rgb210Buffer.gpuBuf()),
    // PixelArray<Image::ConstRGBSolidPixel>(width, height, inputBuffer.gpuBuf()), stream);
    ENSURE(stream.synchronize());
  }

  DeviceBuffer<uchar3> outputBuffer(width, height);
  {
    ENSURE(stream.synchronize());
    ProfilerT p("RGB unpack", false, Logger::get(Logger::Info));
    // packUnpack(PixelArray<Image::RGBSolidPixel>(width, height, outputBuffer.gpuBuf()),
    // PixelArray<Image::ConstRGB210Pixel>(width, height, rgb210Buffer.gpuBuf()), stream);
    ENSURE(stream.synchronize());
  }

  std::vector<unsigned char> output;
  outputBuffer.readbackData(output);

  for (unsigned i = 0; i < width * height; ++i) {
    ENSURE_EQ((int)input[3 * i + 0], (int)output[3 * i + 0]);
    ENSURE_EQ((int)input[3 * i + 1], (int)output[3 * i + 1]);
    ENSURE_EQ((int)input[3 * i + 2], (int)output[3 * i + 2]);
  }
}

template <class ProfilerT>
void testYV12Minimal(unsigned width, unsigned height) {
  ENSURE(!(width & 1), "width must be even");
  ENSURE(!(height & 1), "height must be even");
  // Generate YV12 data
  unsigned char* input = new unsigned char[3 * width * height / 2];
  for (unsigned i = 0; i < width * height; ++i) {
    input[i] = 124;
  }
  for (unsigned i = 0; i < width * height / 4; ++i) {
    input[width * height + i] = 94;
  }
  for (unsigned i = 0; i < width * height / 4; ++i) {
    input[width * height + width * height / 4 + i] = 196;
  }

  auto buffer = GPU::uniqueBuffer<unsigned char>(width * height * 3 / 2, "unpackTest");
  ENSURE(buffer.status(), "no device memory");
  ENSURE(GPU::memcpyBlocking(buffer.borrow(), input), "transfer error");

  // to internal format
  auto surf = Core::OffscreenAllocator::createSourceSurface(width, height, "testYV12Minimal");
  ENSURE(surf.ok());
  convertYV12ToRGBA(*surf->pimpl->surface, buffer.borrow(), width, height, GPU::Stream::getDefault());

  // from internal format
  auto yOut = GPU::Buffer2D::allocate(width, height, "YV12 Y plane").value();
  auto uOut = GPU::Buffer2D::allocate(width / 2, height / 2, "YV12 U plane").value();
  auto vOut = GPU::Buffer2D::allocate(width / 2, height / 2, "YV12 V plane").value();
  unpackYV12(yOut, uOut, vOut, *surf->pimpl->surface, width, height, GPU::Stream::getDefault());

  // Check result of conversion
  std::vector<unsigned char> outputY(width * height);
  ENSURE(GPU::memcpyBlocking(outputY.data(), yOut), "transfer error");
  std::vector<unsigned char> outputU(width * height / 4);
  ENSURE(GPU::memcpyBlocking(outputU.data(), uOut), "transfer error");
  std::vector<unsigned char> outputV(width * height / 4);
  ENSURE(GPU::memcpyBlocking(outputV.data(), vOut), "transfer error");

  int eps = 1;
  for (unsigned i = 0; i < width * height; ++i) {
    ENSURE_APPROX_EQ(124, (int)outputY[i], eps);
  }
  for (unsigned i = 0; i < (width * height) / 4; ++i) {
    ENSURE_APPROX_EQ(94, (int)outputU[i], eps);
  }
  for (unsigned i = 0; i < (width * height) / 4; ++i) {
    ENSURE_APPROX_EQ(196, (int)outputV[i], eps);
  }

  // clean up
  delete[] input;
  ENSURE(yOut.release());
  ENSURE(uOut.release());
  ENSURE(vOut.release());
}

template <class ProfilerT>
void testNV12Minimal(unsigned width, unsigned height) {
  ENSURE(!(width & 1), "width must be even");
  ENSURE(!(height & 1), "height must be even");

  // Generate NV12 data
  std::vector<unsigned char> input(3 * width * height / 2);
  for (unsigned i = 0; i < width * height; ++i) {
    input[i] = 80 + i % 48;
  }
  for (unsigned i = 0; i < width * height / 2; i += 2) {
    input[width * height + i] = 90 + (i / width) % 80;
    input[width * height + i + 1] = 80 + (i / height) % 100;
  }

  // alloc
  auto buffer = GPU::uniqueBuffer<unsigned char>(width * height * 3 / 2, "unpackTest");
  ENSURE(buffer.status(), "no device memory");

  // transfer
  ENSURE(GPU::memcpyBlocking(buffer.borrow(), input.data()), "transfer error");

  // to internal format
  auto surf = Core::OffscreenAllocator::createSourceSurface(width, height, "testNV12Minimal");
  ENSURE(surf.ok());
  convertNV12ToRGBA(*surf->pimpl->surface, buffer.borrow(), width, height, GPU::Stream::getDefault());

#ifdef DUMP_IMAGES
  GPU::Stream::getDefault().synchronize();
  Debug::dumpRGBATexture("/tmp/nv12.png", *surf->pimpl->surface, width, height);
#endif

  // from internal format
  GPU::Buffer2D yOut = GPU::Buffer2D::allocate(width, height, "NV12 Y plane").value();
  GPU::Buffer2D uvOut = GPU::Buffer2D::allocate(width, height / 2, "NV12 UV plane").value();
  unpackNV12(yOut, uvOut, *surf->pimpl->surface, width, height, GPU::Stream::getDefault());

  // Check result of conversion
  std::vector<unsigned char> outputY(width * height);
  ENSURE(GPU::memcpyBlocking(outputY.data(), yOut), "transfer error");
  std::vector<unsigned char> outputUV(width * height / 2);
  ENSURE(GPU::memcpyBlocking(outputUV.data(), uvOut), "transfer error");

  GPU::Stream::getDefault().synchronize();

  int eps = 1;
  for (unsigned i = 0; i < width * height; ++i) {
    ENSURE_APPROX_EQ((int)(80 + i % 48), (int)outputY[i], eps);
  }
  for (unsigned i = 0; i < width * height / 2; i += 2) {
    ENSURE_APPROX_EQ((int)(90 + (i / width) % 80), (int)outputUV[i], eps);
    ENSURE_APPROX_EQ((int)(80 + (i / height) % 100), (int)outputUV[i + 1], eps);
  }

  // clean up
  ENSURE(yOut.release());
  ENSURE(uvOut.release());
}

// TODO_OPENCL_IMPL
#ifndef VS_OPENCL

void testBayerRGGB() {
  const int width = 6;
  const int height = 4;
  DeviceBuffer<unsigned char> bayerFiltered(width, height);
  const std::vector<unsigned char> inputData = {
      10, 42, 36, 78, 40, 52, 50, 64, 8, 82, 42, 44, 90, 112, 124, 122, 162, 56, 130, 144, 16, 48, 44, 128,
  };
  bayerFiltered.fill(inputData);

  PackedDeviceBuffer outputDev(width, height);
  convertBayerRGGBToRGBA(outputDev.gpuBuf(), bayerFiltered.gpuBufConst(), width, height, GPU::Stream::getDefault());

  std::vector<uint32_t> actual;
  outputDev.readback(actual);
  /*
  std::cout << "R" << std::endl;
  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x) {
      std::cout << RGBA::r(actual[y * width + x]) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "G" << std::endl;
  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x) {
      std::cout << RGBA::g(actual[y * width + x]) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "B" << std::endl;
  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x) {
      std::cout << RGBA::b(actual[y * width + x]) << " ";
    }
    std::cout << std::endl;
  }
  */

  // Red:
  //
  //   10,    ,  36,    ,  40,    ,                                                                      10,   23,   36,
  //   38,   40,   40, (40)
  //     ,    ,    ,    ,    ,    ,    =>                                                    =>          50,   65,   80,
  //     90,  101,  101, (XX)
  //   90,    , 124,    , 162,    ,                                                                      90,  107,  124,
  //   143,  162,  162, (162)
  //     ,    ,    ,    ,    ,    ,                                                                      90,  107,  124,
  //     143,  162,  162, (XX)
  //                                                                                                   (90), (XX),(124),
  //                                                                                                   (XX),(162), (XX),
  //                                                                                                   (162)

  // Green:
  //                                         (XX), (50), (XX),  (8), (XX), (42), (XX), (XX)      (XX), (50), (XX),  (8),
  //                                         (XX), (42), (XX), (XX)
  //     ,  42,    ,  78,    ,  52,          (42),     ,   42,     ,   78,     ,   52, (XX)      (42),   46,   42,   34,
  //     78,   53,   52, (XX)
  //   50,    ,   8,    ,  42,    ,     =>   (XX),   50,     ,    8,     ,   42,     , (42)  =>  (XX),   50,   53,    8,
  //   62,   42,   48, (42)
  //     , 112,    , 122,    ,  56,         (112),     ,  112,     ,  122,     ,   56, (XX)     (112),  101,  112,   64,
  //     122,   66,   56, (XX)
  //  130,    ,  16,    ,  44,    ,          (XX),  130,     ,   16,     ,   44,     , (44)      (XX),  130,   92,   16,
  //  76,   44,   50, (44)
  //                                         (XX), (XX),(112), (XX),(122), (XX), (56), (XX)      (XX), (XX),(112),
  //                                         (XX),(122), (XX), (56), (XX)

  // Blue:
  //                                                                                             (64), (XX), (64), (XX),
  //                                                                                             (82), (XX), (44),
  //     ,    ,    ,    ,    ,    ,                                                              (XX),   64,   64,   73,
  //     82,   63,   44, ,  64,    ,  82,    ,  44,    =>                                                    =>  (64),
  //     64,   64,   73,   82,   63,   44, ,    ,    ,    ,    ,    , (XX),  104,  104,   84,   65,   75,   86, , 144,
  //     ,  48,    , 128,                                                             (144),  144,  144,   96,   48, 88,
  //     128,

  const std::vector<uint32_t> expected = {
      RGBA::pack(10, 46, 64, 255),   RGBA::pack(23, 42, 64, 255),    RGBA::pack(36, 34, 73, 255),
      RGBA::pack(38, 78, 82, 255),   RGBA::pack(40, 53, 63, 255),    RGBA::pack(40, 52, 44, 255),
      RGBA::pack(50, 50, 64, 255),   RGBA::pack(65, 53, 64, 255),    RGBA::pack(80, 8, 73, 255),
      RGBA::pack(90, 62, 82, 255),   RGBA::pack(101, 42, 63, 255),   RGBA::pack(101, 48, 44, 255),
      RGBA::pack(90, 101, 104, 255), RGBA::pack(107, 112, 104, 255), RGBA::pack(124, 64, 84, 255),
      RGBA::pack(143, 122, 65, 255), RGBA::pack(162, 66, 75, 255),   RGBA::pack(162, 56, 86, 255),
      RGBA::pack(90, 130, 144, 255), RGBA::pack(107, 92, 144, 255),  RGBA::pack(124, 16, 96, 255),
      RGBA::pack(143, 76, 48, 255),  RGBA::pack(162, 44, 88, 255),   RGBA::pack(162, 50, 128, 255),
  };
  ENSURE_RGBA8888_ARRAY_EQ(expected.data(), actual.data(), width, height);
}

void benchBayerRGGB(const int width, const int height) {
  std::vector<unsigned char> inputData(width * height);
  for (int i = 0; i < width * height; ++i) {
    inputData[i] = (unsigned char)((i * 1357) % 255);
  }

  {
    DeviceBuffer<unsigned char> bayerFiltered(width, height);
    {
      Util::SimpleProfiler p("Debayer bench VS, copy to GPU", false, Logger::get(Logger::Info));
      bayerFiltered.fill(inputData);
    }
    PackedDeviceBuffer vsOutputDev(width, height);
    {
      Util::SimpleProfiler p("Debayer bench VS, debayer", false, Logger::get(Logger::Info));
      convertBayerRGGBToRGBA(vsOutputDev.gpuBuf(), bayerFiltered.gpuBufConst(), width, height,
                             GPU::Stream::getDefault());
      cudaDeviceSynchronize();
    }
    std::vector<uint32_t> vsOutput;
    {
      Util::SimpleProfiler p("Debayer bench VS, readback", false, Logger::get(Logger::Info));
      vsOutputDev.readback(vsOutput);
    }
  }

#ifdef WITH_OPENCV_CMP
  {
    std::unique_ptr<cv::gpu::GpuMat> cvSrc;
    {
      Util::SimpleProfiler p("Debayer bench OpenCV, create image, copy to GPU", false, std::cout);
      cvSrc.reset(new cv::gpu::GpuMat(cv::Mat(height, width, CV_8UC1, (void*)inputData.data())));
    }
    cv::gpu::GpuMat cvDevOutput(height, width, CV_8UC4);
    {
      Util::SimpleProfiler p("Debayer bench OpenCV, debayer", false, std::cout);
      cv::gpu::cvtColor(*cvSrc, cvDevOutput, CV_BayerBG2BGR);
      cudaDeviceSynchronize();
    }
    cv::Mat cvOutput(height, width, CV_8UC4);
    {
      Util::SimpleProfiler p("Debayer bench OpenCV, readback", false, std::cout);
      cvDevOutput.download(cvOutput);
      cudaDeviceSynchronize();
    }
  }
#endif
}

#ifdef WITH_OPENCV_CMP
void testBayerRGGBReal() {
  const int width = 600;
  const int height = 480;

  cv::Mat cvOutput(height, width, CV_8UC4);
  cv::Mat vsOutput(height, width, CV_8UC4);

  {
    DeviceBuffer<unsigned char> bayerFiltered(width, height);
    bayerFiltered.readRawFromFile("data/bayer/mire.gray");
    PackedDeviceBuffer vsOutputDev(width, height);
    {
      Util::SimpleProfiler p("Debayer real VS, debayer", false, std::cout);
      convertBayerRGGBToRGBA(vsOutputDev.ptr(), bayerFiltered.ptr(), width, height, 16);
      cudaDeviceSynchronize();
    }
    std::vector<uint32_t> tmp;
    vsOutputDev.readback(tmp);
    std::unique_ptr<std::ostream> os(Util::PpmWriter::openPpm("debayer-vs.ppm", width, height));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const uint32_t v = tmp[y * width + x];
        *os << (char)RGBA::r(v);
        *os << (char)RGBA::g(v);
        *os << (char)RGBA::b(v);
      }
    }
    vsOutput = cv::Mat(height, width, CV_8UC4, (void*)tmp.data()).clone();
  }
  std::cout << vsOutput.at<uint32_t>(200, 200) << std::endl;

  {
    cv::gpu::GpuMat cvSrc(cv::imread("data/bayer/mire.pgm", CV_LOAD_IMAGE_GRAYSCALE));
    cv::gpu::GpuMat cvDevOutput(height, width, CV_8UC4);
    {
      Util::SimpleProfiler p("Debayer real OpenCV, debayer", false, std::cout);
      cv::gpu::cvtColor(cvSrc, cvDevOutput, CV_BayerBG2BGR);
      cudaDeviceSynchronize();
    }
    cvDevOutput.download(cvOutput);
  }
  cv::imwrite("debayer-opencv.ppm", cvOutput);
  // cv::imwrite("debayer-vs.ppm", vsOutput);
}
#endif

#endif  // VS_OPENCL

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // VideoStitch::Testing::testOutOfPlaceRGB<NoProfiler>(16, 16);
  // VideoStitch::Testing::testOutOfPlaceRGB<NoProfiler>(312, 531);

  // VideoStitch::Testing::testOutOfPlaceMonoY<NoProfiler>(16, 16);
  // VideoStitch::Testing::testOutOfPlaceMonoY<NoProfiler>(312, 531);

  VideoStitch::Testing::testYV12Minimal<NoProfiler>(16, 16);
  VideoStitch::Testing::testYV12Minimal<NoProfiler>(312, 532);

  VideoStitch::Testing::testNV12Minimal<NoProfiler>(16, 16);
  VideoStitch::Testing::testNV12Minimal<NoProfiler>(280, 124);

// TODO_OPENCL_IMPL
#ifndef VS_OPENCL

  // VideoStitch::Testing::testBayerRGGB();

#endif

#ifdef WITH_OPENCV_CMP
  VideoStitch::Testing::testBayerRGGBReal();
#endif

  if (argc > 1 && !strcmp(argv[1], "bench")) {
    VideoStitch::Testing::testOutOfPlaceRGB<VideoStitch::Util::SimpleProfiler>(6542, 5322);

// TODO_OPENCL_IMPL
#ifndef VS_OPENCL
    VideoStitch::Testing::benchBayerRGGB(6542, 5322);
#endif
  }
  VideoStitch::GPU::Context::destroy();
  return 0;
}
