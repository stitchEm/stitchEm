// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/util.hpp"
#include "gpu/testing.hpp"
#include "common/fakeReader.hpp"
#include "common/fakeMerger.hpp"
#include "common/fakeWriter.hpp"

#include <core/stitchOutput/stitchOutput.hpp>
#include "libvideostitch/context.hpp"
#include "libvideostitch/allocator.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/controller.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/processorStitchOutput.hpp"
#include "libvideostitch/parse.hpp"
#include <parallax/noFlow.hpp>
#include <parallax/noWarper.hpp>

#include <cstring>
#include <memory>

// how many miliseconds per time-sensitive testing step?
// increase (e.g. to 1000) if test fails due to being too short
static const int tickLength = 50;

namespace VideoStitch {
namespace Testing {

class MockVideoWriterRace : public Output::VideoWriter {
 public:
  MockVideoWriterRace(const std::string& n, unsigned w, unsigned h, std::atomic<int>* numCalls,
                      VideoStitch::PixelFormat format = PixelFormat::RGBA)
      : Output(n), VideoWriter(w, h, {60, 1}, format), numCalls(numCalls) {
    *numCalls = 0;
  }

  void pushVideo(const Frame& frame) {
    ENSURE(*numCalls == (int)frame.pts);
    ++*numCalls;
  }

  void pushAudio(Audio::Samples&) {}

 private:
  std::atomic<int>* numCalls;
};

Core::PanoDefinition* getTestPanoDef() {
  Potential<Ptv::Parser> parser(Ptv::Parser::create());
  if (!parser->parseData("{"
                         " \"width\": 513, "
                         " \"height\": 315, "
                         " \"hfov\": 90.0, "
                         " \"proj\": \"rectilinear\", "
                         " \"inputs\": [ "
                         "  { "
                         "   \"width\": 17, "
                         "   \"height\": 13, "
                         "   \"hfov\": 90.0, "
                         "   \"yaw\": 0.0, "
                         "   \"pitch\": 0.0, "
                         "   \"roll\": 0.0, "
                         "   \"proj\": \"rectilinear\", "
                         "   \"viewpoint_model\": \"ptgui\", "
                         "   \"response\": \"linear\", "
                         "   \"filename\": \"\" "
                         "  }, "
                         "  { "
                         "   \"width\": 17, "
                         "   \"height\": 13, "
                         "   \"hfov\": 90.0, "
                         "   \"yaw\": 0.0, "
                         "   \"pitch\": 0.0, "
                         "   \"roll\": 0.0, "
                         "   \"proj\": \"rectilinear\", "
                         "   \"viewpoint_model\": \"ptgui\", "
                         "   \"response\": \"linear\", "
                         "   \"filename\": \"\" "
                         "  } "
                         " ]"
                         "}")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return NULL;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);
  return panoDef.release();
}

Core::PotentialController getController() {
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());

  FakeReaderFactory* fakeReaderFactory = new FakeReaderFactory(0);
  std::atomic<int> totalNumSetups(0);
  FakeImageMergerFactory factory(Core::ImageMergerFactory::CoreVersion1, &totalNumSetups);
  factory.setHash("hash1");

  ENSURE_EQ(0, (int)totalNumSetups);
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(Core::AudioPipeDefinition::createDefault());
  return Core::createController(*panoDef, factory, Core::NoWarper::Factory(), Core::NoFlow::Factory(),
                                fakeReaderFactory, *audioPipeDef);
}

// In this test we assume a negligible filling time.
void testInOrder() {
  Core::PotentialController controller = getController();

  std::cout << "*** Test in order ***" << std::endl;
  const unsigned w = 1;
  const unsigned h = 1;
  std::atomic<int> numCalls;
  const_cast<Core::PanoDefinition&>(controller->getPano()).setWidth(1);
  const_cast<Core::PanoDefinition&>(controller->getPano()).setHeight(1);

  std::vector<std::shared_ptr<Core::PanoSurface>> surfaces;
  for (int i = 0; i < 2; ++i) {
    auto surf = Core::OffscreenAllocator::createPanoSurface(1, 1, "StitchOutputTest");
    ENSURE(surf.status());
    surfaces.push_back(std::shared_ptr<Core::PanoSurface>(surf.release()));
  }

  std::shared_ptr<MockVideoWriter> outputWriter(new MockVideoWriter("0", w, h, 1 * tickLength, &numCalls));
  Core::PotentialStitchOutput stitchOutput = controller->createAsyncStitchOutput(surfaces, outputWriter);

  std::cout << "Make sure no filled frames blocks the writing..." << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * tickLength));
  ENSURE(numCalls == 0);

  std::cout << "Make sure filling a frame unblocks it..." << std::endl;
  ENSURE(stitchOutput->pimpl->pushVideo(0));
  ENSURE(numCalls == 0);  // It will still take 2 tickLength's to write
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * tickLength));
  ENSURE(numCalls == 1);  // but eventually the result will be written.

  std::cout << "Make sure a full buffer blocks filling..." << std::endl;
  ENSURE(stitchOutput->pimpl->pushVideo(1));  // 1
  ENSURE(stitchOutput->pimpl->pushVideo(2));  // 2
  std::cout << "  2 frames filled." << std::endl;
  ENSURE(numCalls == 1);  // At this point the writer has still not caught up
  std::cout << "  filling one more, will block." << std::endl;
  ENSURE(stitchOutput->pimpl->pushVideo(
      3));  // 4 This one blocks until the frame corresponding to the first call to fill() above has been written
  std::cout << "  done blocking." << std::endl;
  ENSURE(numCalls == 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(4 * tickLength));
  std::cout << "Make sure destruction waits for pending processing..." << std::endl;
  ENSURE(numCalls == 4);
}

// In this test we assume a negligible filling time.
/* In AsyncBufferedOutput<>::consumerThread, there is no delay between it pushs back a frame in the blankFrames queue
   and it looks for the next frame in the stitchedFrames queue. With a queue size fixed to 2 by the controller it makes
   difficult 2 fill() to occurs without a consumerThread to occurs. Thus, there is almost always only one frame
   waiting in the stitchedFrames queue so that the order in which they are pushed/poped was not really tested.
   This test checks the use of stitchedFrames.pop_front() instead of stitchedFrames.pop_back()
   in AsyncBufferedOutput<>::consumerThread.
   To enable the race condition to occurs, multiples fill() are done in a row. */

void testRaceInOrder() {
  Core::PotentialController controller = getController();

  std::cout << "*** Test Race in order ***" << std::endl;
  const unsigned w = 1;
  const unsigned h = 1;
  std::atomic<int> numCalls;
  const_cast<Core::PanoDefinition&>(controller->getPano()).setWidth(1);
  const_cast<Core::PanoDefinition&>(controller->getPano()).setHeight(1);

  std::vector<std::shared_ptr<Core::PanoSurface>> surfaces;
  for (int i = 0; i < 2; ++i) {
    auto surf = Core::OffscreenAllocator::createPanoSurface(1, 1, "StitchOutputTest");
    ENSURE(surf.status());
    surfaces.push_back(std::shared_ptr<Core::PanoSurface>(surf.release()));
  }

  std::shared_ptr<MockVideoWriterRace> outputWriter(new MockVideoWriterRace("0", w, h, &numCalls));
  Core::PotentialStitchOutput stitchOutput = controller->createAsyncStitchOutput(surfaces, outputWriter);

  const int nbFill = 100;

  /* generates multiples fill() so that sometimes 2 fill() may be processed between
    2 AsyncBufferedOutput<>::consumerThread() */
  std::thread fullFill([&]() {
    for (int k = 0; k < nbFill; k++) {
      ENSURE(stitchOutput->pimpl->pushVideo(k));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2 * tickLength));
  });

  fullFill.join();
  std::cout << "Make sure destruction waits for pending processing..." << std::endl;
  ENSURE(numCalls == nbFill);
}

void testTeeResults() {
  std::cout << "*** Test tee result ***" << std::endl;
  Core::PotentialController controller = getController();

  const unsigned w = 2;
  const unsigned h = 2;
  const_cast<Core::PanoDefinition&>(controller->getPano()).setWidth(2);
  const_cast<Core::PanoDefinition&>(controller->getPano()).setHeight(2);
  Audio::Samples audioSamples;

  std::vector<std::shared_ptr<Core::PanoSurface>> surfaces;
  for (int i = 0; i < 2; ++i) {
    auto surf = Core::OffscreenAllocator::createPanoSurface(2, 2, "StitchOutputTest");
    ENSURE(surf.status());
    surfaces.push_back(std::shared_ptr<Core::PanoSurface>(surf.release()));
  }
  std::vector<std::shared_ptr<Output::VideoWriter>> outputWriters;
  outputWriters.emplace_back(new MockVideoWriter2("0", w, h, VideoStitch::PixelFormat::RGBA));
  outputWriters.emplace_back(new MockVideoWriter2("1", w, h, VideoStitch::PixelFormat::RGB));
  outputWriters.emplace_back(new MockVideoWriter2("2", w, h, VideoStitch::PixelFormat::YV12));
  std::vector<std::shared_ptr<Core::PanoRenderer>> renderers;
  Core::PotentialStitchOutput stitchOutput = controller->createAsyncStitchOutput(surfaces, renderers, outputWriters);

  std::vector<uint32_t> panoDevData;
  panoDevData.push_back(Image::RGBA::pack(42, 255, 87, 0xff));
  panoDevData.push_back(Image::RGBA::pack(42, 255, 87, 0xff));
  panoDevData.push_back(Image::RGBA::pack(42, 255, 87, 0xff));
  panoDevData.push_back(Image::RGBA::pack(42, 255, 87, 0xff));
  Core::PanoSurface& panorama = stitchOutput->pimpl->acquireFrame(0);
  ENSURE(GPU::memcpyBlocking(panorama.pimpl->buffer, panoDevData.data()));

  stitchOutput->pimpl->pushVideo(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ENSURE_EQ((size_t)(w * h * 4), std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[0])->lastFrameData().size());
  for (int64_t i = 0; i < w * h; ++i) {
    ENSURE_EQ(42, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[0])->lastFrameData()[4 * i + 0]);
    ENSURE_EQ(255, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[0])->lastFrameData()[4 * i + 1]);
    ENSURE_EQ(87, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[0])->lastFrameData()[4 * i + 2]);
    ENSURE_EQ(255, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[0])->lastFrameData()[4 * i + 3]);
  }

  ENSURE_EQ((size_t)(w * h * 3), std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[1])->lastFrameData().size());
  for (int64_t i = 0; i < w * h; ++i) {
    ENSURE_EQ(42, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[1])->lastFrameData()[3 * i + 0]);
    ENSURE_EQ(255, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[1])->lastFrameData()[3 * i + 1]);
    ENSURE_EQ(87, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[1])->lastFrameData()[3 * i + 2]);
  }

  ENSURE_EQ((size_t)((w * h * 3) / 2),
            std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData().size());
  ENSURE_EQ(164, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData()[0]);
  ENSURE_EQ(164, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData()[1]);
  ENSURE_EQ(164, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData()[2]);
  ENSURE_EQ(164, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData()[3]);
  ENSURE_EQ(86, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData()[4]);
  ENSURE_EQ(47, (int)std::dynamic_pointer_cast<MockVideoWriter2>(outputWriters[2])->lastFrameData()[5]);
}

void testSumProcessorStitchOutput() {
  const int64_t width = 5;
  const int64_t height = 5;

  Potential<Core::ProcessorStitchOutput> stitchOutput(
      Core::ProcessorStitchOutput::create(width, height, Core::ProcessorStitchOutput::Spec().withSum().withCount()));

  Audio::Samples audioSamples;

  ENSURE(stitchOutput.status());

  std::vector<uint32_t> inputData(width * height);
  inputData[0] = Image::RGBA::pack(42, 255, 87, 0xff);
  inputData[1] = Image::RGBA::pack(255, 255, 255, 0xff);

  Core::PanoSurface& pano = stitchOutput->pimpl->acquireFrame(0);
  ENSURE(GPU::memcpyBlocking(pano.pimpl->buffer, inputData.data()));

  ENSURE(stitchOutput->pimpl->pushVideo(0));
  ENSURE(stitchOutput->getResult().has("sum"), "missing 'sum' field");
  ENSURE(stitchOutput->getResult().has("count"), "missing 'count' field");
  int64_t expectedSum = (42 + 255 + 87) / 3 + (255 + 255 + 255) / 3;
  ENSURE_EQ(expectedSum, stitchOutput->getResult().has("sum")->asInt());
  ENSURE_EQ(2, (int)stitchOutput->getResult().has("count")->asInt());
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // VideoStitch::Testing::testInOrder();
  // VideoStitch::Testing::testRaceInOrder();

  VideoStitch::Testing::testTeeResults();

// TODO_OPENCL_IMPL
#ifndef VS_OPENCL
  // postprocessor
  VideoStitch::Testing::testSumProcessorStitchOutput();
#endif

  VideoStitch::GPU::Context::destroy();
  return 0;
}
