// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <gpu/memcpy.hpp>
#include <mask/mergerMask.hpp>
#include <mask/seamFinder.hpp>
#include <mask/mergerMaskConstant.hpp>
#include <util/pngutil.hpp>
#include <util/imageProcessingGPUUtils.hpp>

#include <string>
#include <sstream>

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
//#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "../util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Testing {

GPU::UniqueBuffer<uint32_t> loadFile(const char *filename, int64_t &width, int64_t &height,
                                     const bool colorConversion = false) {
  std::vector<unsigned char> tmp(width * height * 4);
  ENSURE(VideoStitch::Util::PngReader::readRGBAFromFile(filename, width, height, &tmp[0]), "Read RGBA image from file");

  std::vector<uint32_t> buffer((size_t)(width * height));
  for (size_t i = 0; i < (size_t)(width * height); ++i) {
    if (tmp[(size_t)(4 * i + 3)] == 0) {
      buffer[i] = INVALID_VALUE;
    } else {
      if (!colorConversion) {
        buffer[i] = VideoStitch::Image::RGBA::pack(tmp[(size_t)(4 * i)], tmp[(size_t)(4 * i + 1)],
                                                   tmp[(size_t)(4 * i + 2)], tmp[(size_t)(4 * i + 3)]);
      } else {
        buffer[i] = VideoStitch::Image::RGB210::pack(tmp[(size_t)(4 * i)], tmp[(size_t)(4 * i + 1)],
                                                     tmp[(size_t)(4 * i + 2)], tmp[(size_t)(4 * i + 3)]);
      }
    }
  }

  auto devBuffer = GPU::uniqueBuffer<uint32_t>((size_t)(width * height), "SeamTest");
  ENSURE(devBuffer.status());

  if (colorConversion) {
    auto devOut = GPU::uniqueBuffer<uint32_t>((size_t)(width * height), "SeamTest");
    ENSURE(devOut.status());
    auto workBuffer = GPU::uniqueBuffer<uint32_t>((size_t)(width * height), "SeamTest");
    ENSURE(workBuffer.status());
    auto blurBuffer = GPU::uniqueBuffer<uint32_t>((size_t)(width * height), "SeamTest");
    ENSURE(blurBuffer.status());
    ENSURE(GPU::memcpyBlocking(devOut.borrow(), &buffer.front()));
    ENSURE(Util::ImageProcessingGPU::convertRGB210ToRGBandGradient(make_int2((int)width, (int)height), devOut.borrow(),
                                                                   devBuffer.borrow(), GPU::Stream::getDefault()));
    ENSURE(Util::ImageProcessingGPU::convertRGBandGradientToNormalizedLABandGradient(
        make_int2((int)width, (int)height), devBuffer.borrow(), GPU::Stream::getDefault()));
  } else {
    ENSURE(GPU::memcpyBlocking(devBuffer.borrow(), &buffer.front()));
  }
  return devBuffer.releaseValue();
}

std::string myreplace(std::string &s, const std::string &toReplace, const std::string &replaceWith) {
  return (s.replace(s.find(toReplace), toReplace.length(), replaceWith));
}

void testSeam() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif

  std::vector<std::pair<std::string, std::string>> seamTests;
  std::vector<std::string> seamResults;

  for (int i = 0; i <= 19; i++) {
    seamTests.push_back(std::make_pair(workingPath + "data/seam/test" + std::to_string(i) + "-a.png",
                                       workingPath + "data/seam/test" + std::to_string(i) + "-b.png"));
    seamResults.push_back(workingPath + "data/seam/test" + std::to_string(i) + "-result.png");
  }

  // Original test image size, can be downloaded from \\nas_vs\Assets\VideoStitch-assets\seam
  /*std::vector<int2> sizes = {
    make_int2(960, 480), make_int2(1920, 960), make_int2(504, 360), make_int2(400, 400), make_int2(512, 512),
    make_int2(400, 400), make_int2(960, 480), make_int2(3840, 1920), make_int2(1396, 698), make_int2(1024, 512),
    make_int2(4096, 2048), make_int2(3840, 1920), make_int2(400, 400), make_int2(1920, 960), make_int2(1920, 960),
    make_int2(1024, 512), make_int2(2048, 1024), make_int2(1920, 960), make_int2(2048, 1024), make_int2(2048, 1024) };*/

  std::vector<int2> sizes = {make_int2(100, 50), make_int2(100, 50), make_int2(100, 71), make_int2(50, 50),
                             make_int2(50, 50),  make_int2(50, 50),  make_int2(100, 50), make_int2(100, 50),
                             make_int2(100, 50), make_int2(100, 50), make_int2(100, 50), make_int2(100, 50),
                             make_int2(50, 50),  make_int2(100, 50), make_int2(100, 50), make_int2(100, 50)};

  for (int i = 5; i >= 4; i--) {
    int64_t width = sizes[i].x;
    int64_t height = sizes[i].y;
    const std::string fileA = seamTests[i].first;
    const std::string fileB = seamTests[i].second;
    auto devBuffer0 = VideoStitch::Testing::loadFile(fileA.c_str(), width, height, true);
    auto devBuffer1 = VideoStitch::Testing::loadFile(fileB.c_str(), width, height, true);
#ifdef DUMP_TEST_RESULT

    /*{
      std::stringstream ss;
      ss << workingPath << "data/seam/test" << i << "-gradient-0.png";
      auto gradientOut = GPU::uniqueBuffer<unsigned char>((size_t)(width * height), "SeamTest");
      ENSURE(MergerMask::MergerMask::extractChannel(make_int2(width, height), devBuffer0.borrow(), 3,
    gradientOut.borrow(), GPU::Stream::getDefault())); Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(),
    gradientOut.borrow(), width, height);
    }
    {
      std::stringstream ss;
      ss << workingPath << "data/seam/test" << i << "-gradient-1.png";
      auto gradientOut = GPU::uniqueBuffer<unsigned char>((size_t)(width * height), "SeamTest");
      ENSURE(MergerMask::MergerMask::extractChannel(make_int2(width, height), devBuffer1.borrow(), 3,
    gradientOut.borrow(), GPU::Stream::getDefault())); Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(),
    gradientOut.borrow(), width, height);
    }*/
#endif

    Core::Rect rect = Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height - 1, width - 1);
    GPU::Stream stream = VideoStitch::GPU::Stream::getDefault();

    auto potSeamFinder = MergerMask::SeamFinder::create(2, 1, (int)width, rect, devBuffer0.borrow().as_const(), rect,
                                                        devBuffer1.borrow().as_const(), stream);
    ENSURE(potSeamFinder.status());
    std::unique_ptr<MergerMask::SeamFinder> seamFinder(potSeamFinder.release());

    ENSURE(seamFinder->findSeam());
    // If there is no overlapping, just continue the process
    if (seamFinder->seamFound()) {
      auto devOriBuffer0 = VideoStitch::Testing::loadFile(fileA.c_str(), width, height, false);
      auto devOriBuffer1 = VideoStitch::Testing::loadFile(fileB.c_str(), width, height, false);
      ENSURE(seamFinder->replaceBuffers(devOriBuffer0.borrow().as_const(), devOriBuffer1.borrow().as_const()));

#ifdef DUMP_TEST_RESULT
      // This part is used to dump the result
      {
        std::stringstream ss;
        ss << workingPath << "data/seam/test" << i << "-result.png";
        ENSURE(seamFinder->saveSeamImage(ss.str(), 2));
      }
      {
        const int id = 1;
        std::vector<int> components;
        ENSURE(seamFinder->findConnectedComponentsAfterCuts(id, components));
        std::stringstream ss;
        ss << workingPath << "data/seam/test" << i << "-component.png";
        Debug::dumpRGBAIndexDeviceBuffer<int>(ss.str().c_str(), components, width, height);
      }
      // Dump the outputs map index
      {
        std::stringstream ss;
        ss.str("");
        ss << workingPath << "data/seam/test" << i << "-outputsmap.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), seamFinder->getOutputsMap().as_const(), width, height);
      }
      // Dump the outputs map index
      {
        std::stringstream ss;
        ss.str("");
        ss << workingPath << "data/seam/test" << i << "-inputsmap.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), seamFinder->getInputsMap().as_const(), width, height);
      }
#else
      std::vector<unsigned char> data;
      seamFinder->saveSeamToBuffer(2, data);

      const std::string fileResult = seamResults[i];
      ENSURE_PNG_FILE_EQ(fileResult, data);
      std::cout << "*** Test " << i << " passed." << std::endl;
#endif
    } else {
      std::cout << "*** Test " << i << " failed." << std::endl;
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testSeam();

  return 0;
}
