// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/surface.hpp"
#include "gpu/testing.hpp"

#include "backend/common/core/types.hpp"

#include "common/fakeReader.hpp"
#include "common/ptv.hpp"

#include "core/geoTransform.hpp"
#include "core1/bounds.hpp"
#include "core1/imageMapping.hpp"
#include "core1/inputsMap.hpp"
#include "gpu/core1/transform.hpp"
#include "gpu/memcpy.hpp"

#include "parallax/flowConstant.hpp"
#include "parallax/mergerPair.hpp"
#include "util/opticalFlowUtils.hpp"
#include "util/pngutil.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/imageProcessingUtils.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/gpu_device.hpp"

#include <algorithm>
#include <sstream>
#include <string>

#define LARGE_ROOT 1000.0f
/*
 * This test could be used to debug the precomputed coordinate buffers in the Transform class
 */

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
//#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include <util/debugUtils.hpp>
#include <util/opticalFlowUtils.hpp>
#endif

namespace VideoStitch {
namespace Testing {

float wrapDifference(const float x0, const float x1, const int wrapWidth) {
  const float z0 = (float)std::fmod(x0 + wrapWidth / 2 + 10 * wrapWidth, wrapWidth);
  const float z1 = (float)std::fmod(x1 + wrapWidth / 2 + 10 * wrapWidth, wrapWidth);
  const float y0 = std::min<float>(z0, z1);
  const float y1 = std::max<float>(z0, z1);
  return std::min<float>(std::abs(y0 - y1), std::abs(y0 + wrapWidth - y1));
}

void testCoordTransform() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif

  std::vector<std::string> coordTransformTests;
  for (int i = 0; i <= 4; i++) {
    coordTransformTests.push_back(workingPath + "data/ptv/test" + std::to_string(i) + ".ptv");
  }
  GPU::Stream stream = GPU::Stream::getDefault();
  const float threshold = 10.0f;
  for (int test = 3; test >= 0; test--) {
    std::string ptvFile = coordTransformTests[test];
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    ENSURE(parser.ok());
    // Load the project and parse it
    ENSURE(parser->parse(ptvFile));
    ENSURE(parser->getRoot().has("pano"));
    // Create a runtime panorama from the parsed project.
    std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(*parser->getRoot().has("pano")));
    const int scaleFactor = panoDef->getBlendingMaskInputScaleFactor();
    int2 size = make_int2((int)panoDef->getWidth(), (int)panoDef->getHeight());
    std::map<int, std::unique_ptr<Core::Transform>> transforms;
    ENSURE(prepareTransforms(*panoDef.get(), transforms));

    GPU::UniqueBuffer<float2> panoCoord;
    ENSURE(panoCoord.alloc(size.x * size.y, "Coord Transform Test"));
    auto tex = Core::OffscreenAllocator::createCoordSurface(size.x, size.y, "Coord Transform Test");
    ENSURE(tex.ok());
    Core::SourceSurface* devCoord = tex.release();
    Core::Rect outputBounds = Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, size.y - 1, size.x - 1);
    for (auto& transform : transforms) {
      const Core::InputDefinition& inputDef = panoDef->getInput(transform.first);

      GPU::UniqueBuffer<float2> inputCoord;
      const int2 inputSize = make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight());

      ENSURE(inputCoord.alloc(inputSize.x * inputSize.y * scaleFactor * scaleFactor, "Coord Transform Test"));

      ENSURE(transform.second->mapCoordInput(0, scaleFactor, inputCoord.borrow(), *panoDef, inputDef, stream));
      std::vector<float2> inputValues(inputSize.x * inputSize.y * scaleFactor * scaleFactor);
      ENSURE(GPU::memcpyBlocking<float2>(&inputValues[0], inputCoord.borrow_const()));

      ENSURE(transform.second->mapBufferCoord(0, *devCoord->pimpl->surface, outputBounds, *panoDef, inputDef, stream));
      std::vector<float2> outputValues(size.x * size.y);
      ENSURE(GPU::memcpyBlocking(panoCoord.borrow(), *devCoord->pimpl->surface));
      ENSURE(GPU::memcpyBlocking<float2>(&outputValues[0], panoCoord.borrow_const()));

#ifdef DUMP_TEST_RESULT
      {
        std::stringstream ss;
        ss << workingPath << "data/ptv/inputsMap-inputcoord-" << transform.first << ".png";
        GPU::UniqueBuffer<uint32_t> dst;
        ENSURE(dst.alloc(inputDef.getWidth() * inputDef.getHeight() * scaleFactor * scaleFactor, "Merger Mask"));
        ENSURE(Util::OpticalFlow::convertFlowToRGBA(
            make_int2((int)inputDef.getWidth() * scaleFactor, (int)inputDef.getHeight() * scaleFactor),
            inputCoord.borrow_const(), make_int2((int)panoDef->getWidth(), (int)panoDef->getHeight()), dst.borrow(),
            stream));
        ENSURE(Debug::dumpRGBADeviceBuffer(ss.str().c_str(), dst.borrow_const(), inputDef.getWidth() * scaleFactor,
                                           inputDef.getHeight() * scaleFactor));
      }
      {
        std::stringstream ss;
        ss << workingPath << "data/ptv/inputsMap-outputcoord-" << transform.first << ".png";
        GPU::UniqueBuffer<uint32_t> dst;
        ENSURE(dst.alloc(size.x * size.y, "Merger Mask"));
        ENSURE(Util::OpticalFlow::convertFlowToRGBA(size, panoCoord.borrow_const(),
                                                    make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight()),
                                                    dst.borrow(), stream));
        ENSURE(Debug::dumpRGBADeviceBuffer(ss.str().c_str(), dst.borrow_const(), size.x, size.y));
      }
#endif
      const int marginSize = 10;
      for (int i = marginSize; i < inputSize.x - marginSize; i++) {
        for (int j = marginSize; j < inputSize.y - marginSize; j++) {
          const float2 coordInOutput = inputValues[j * inputSize.x * scaleFactor + i];
          if (std::round(coordInOutput.x) < size.x && std::round(coordInOutput.y) < size.y &&
              std::round(coordInOutput.x) >= 0 && std::round(coordInOutput.y) >= 0) {
            const float2 coordInInput =
                outputValues[static_cast<int>(std::round(coordInOutput.y) * size.x + std::round(coordInOutput.x))];
            if (coordInInput.x != INVALID_FLOW_VALUE) {  // To prevent near boundary condition
              bool isBijective = ((std::abs(coordInInput.x - (float(i) / scaleFactor)) < threshold) &&
                                  (std::abs(coordInInput.y - (float(j) / scaleFactor)) < threshold));
              if (!isBijective) {
                std::cout << "*** Test " << test << " failed: " << coordInInput.x << "!=" << float(i) / scaleFactor
                          << " && " << coordInInput.y << "!=" << float(j) / scaleFactor << std::endl;
              }
              ENSURE(isBijective);
            }
          }
        }
      }
    }
    delete devCoord;
    std::cout << "*** Test " << test << " passed." << std::endl;
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testCoordTransform();

  return 0;
}
