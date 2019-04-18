// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "gpu/util.hpp"

#include <gpu/memcpy.hpp>
#include <parallax/imageFlow.hpp>
#include <parallax/mergerPair.hpp>
#include <parallax/imageWarper.hpp>
#include <util/pngutil.hpp>
#include <util/opticalFlowUtils.hpp>
#include <util/imageProcessingGPUUtils.hpp>
#include "libvideostitch/imageProcessingUtils.hpp"
#include "libvideostitch/gpu_device.hpp"

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

void testFlow() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath = "C:/Users/nlz/Code/lol/";
#else
  std::string workingPath = "";
#endif

  std::vector<std::pair<std::string, std::string>> flowTests;
  std::vector<std::string> flowResults;

  for (int i = 0; i <= 0; i++) {
    flowTests.push_back(std::make_pair(workingPath + "data/flow/test" + std::to_string(i) + "-a.png",
                                       workingPath + "data/flow/test" + std::to_string(i) + "-b.png"));
    flowResults.push_back(workingPath + "data/flow/test" + std::to_string(i) + "-result.png");
  }

  for (int i = 0; i >= 0; i--) {
    GPU::Stream stream = GPU::Stream::getDefault();
    int64_t width0, height0, width1, height1;
    const std::string fileA = flowTests[i].first;
    const std::string fileB = flowTests[i].second;
    auto devBuffer0 = VideoStitch::Testing::loadFile(fileA.c_str(), width0, height0);
    ENSURE(devBuffer0.status());
    auto devBuffer1 = VideoStitch::Testing::loadFile(fileB.c_str(), width1, height1);
    ENSURE(devBuffer1.status());
    ENSURE(width0 == width1 && height0 == height1, "Input sizes do not match");
    ENSURE(width0 * height0 * width1 * height1 > 0, "One of the image size is 0");

    // This is a dummy merger pair, it transforms the input into identity
    std::shared_ptr<Core::MergerPair> mergerPair = std::shared_ptr<Core::MergerPair>(
        new Core::MergerPair(-1, 128, (int)width0, (int)height0, 0, 0, devBuffer0.borrow_const(), (int)width1,
                             (int)height1, 0, 0, devBuffer1.borrow_const(), stream));
    Potential<Core::ImageFlow> flow = Core::ImageFlow::factor(Core::ImageFlow::ImageFlowAlgorithm::SimpleFlow,
                                                              mergerPair, std::map<std::string, float>());
    ENSURE(flow.status());

    // Find the flow in a multi-scale manner
    ENSURE(flow->findMultiScaleImageFlow(0, 0, make_int2((int)width0, (int)height0), devBuffer0.borrow(),
                                         make_int2((int)width1, (int)height1), devBuffer1.borrow(), stream));

    auto devOutput = GPU::uniqueBuffer<uint32_t>((size_t)(width0 * height0), "FlowTest");
    ENSURE(devOutput.status());

    // Lookup the warped image using the flow coordinate
    ENSURE(Util::OpticalFlow::coordLookup((int)width0, (int)height0, flow->getFinalFlowBuffer(), (int)width1,
                                          (int)height1, devBuffer1.borrow(), devOutput.borrow(), stream));

    std::stringstream ss;
    ss.str("");
    ss << workingPath + "data/flow/test" + std::to_string(i) + "-lookup.png";
#ifdef DUMP_TEST_RESULT
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devOutput.borrow(), width0, height0);
#else
    ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(ss.str(), devOutput.borrow());
#endif

    // Create the image warper --> time to use the warped image for stitching
    Potential<Core::ImageWarper> wraper =
        Core::ImageWarper::factor(Core::ImageWarper::ImageWarperAlgorithm::LinearFlowWarper, mergerPair,
                                  std::map<std::string, float>(), GPU::Stream::getDefault());

    int2 lookupOffset = flow->getLookupOffset(0);
    GPU::UniqueBuffer<float4> debug;
    ENSURE(debug.alloc(width0 * height0, "FlowTest"));
    GPU::UniqueBuffer<uint32_t> flowWarpedBuffer;
    ENSURE(flowWarpedBuffer.alloc(width0 * height0, "FlowTest"));
    GPU::UniqueBuffer<uint32_t> devOut;
    ENSURE(devOut.alloc(width0 * height0, "FlowTest"));

    // Need color remapping here as well, but this remain for later

    ENSURE(wraper->warp(devOut.borrow(), devBuffer1.borrow(), flow->getExtrapolatedFlowRect(0),
                        flow->getFinalExtrapolatedFlowBuffer(), lookupOffset.x, lookupOffset.y, debug.borrow(),
                        flowWarpedBuffer.borrow(), stream));
    {
      std::stringstream ss;
      ss.str("");
      ss << workingPath + "data/flow/test" + std::to_string(i) + "-warpOut.png";
#ifdef DUMP_TEST_RESULT
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devOut.borrow_const(), width0, height0);
#else
      ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(ss.str(), devOut.borrow());
#endif
    }

    // Dump the blend of the warped pair
    {
      Util::ImageProcessingGPU::buffer2DRGBACompactBlendOffsetOperator(
          Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height0 - 1, width0 - 1), flowWarpedBuffer.borrow(), 0.5f,
          Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height0 - 1, width0 - 1), devBuffer0.borrow_const(), 0.5f,
          Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height1 - 1, width1 - 1), devOut.borrow_const(), stream);
      std::stringstream ss;
      ss.str("");
      ss << workingPath + "data/flow/test" + std::to_string(i) + "-blendWarpOut.png";
#ifdef DUMP_TEST_RESULT
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), flowWarpedBuffer.borrow_const(), width0, height0);
#else
      ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(ss.str(), flowWarpedBuffer.borrow());
#endif
    }

    // Dump the blend of the original pair
    {
      Util::ImageProcessingGPU::buffer2DRGBACompactBlendOffsetOperator(
          Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height0 - 1, width0 - 1), flowWarpedBuffer.borrow(), 0.5f,
          Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height0 - 1, width0 - 1), devBuffer0.borrow_const(), .5f,
          Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, height1 - 1, width1 - 1), devBuffer1.borrow_const(),
          stream);
      std::stringstream ss;
      ss.str("");
      ss << workingPath + "data/flow/test" + std::to_string(i) + "-blendOriginalOut.png";
#ifdef DUMP_TEST_RESULT
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), flowWarpedBuffer.borrow_const(), width0, height0);
#else
      ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(ss.str(), flowWarpedBuffer.borrow());
#endif
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  VideoStitch::Testing::testFlow();

  return 0;
}
