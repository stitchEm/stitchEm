// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "common/fakePtvReader.hpp"

#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/profile.hpp"

#include <core/readerController.hpp>

// how many milliseconds per time-sensitive testing step?
// increase (e.g. to 100) if test fails due to being too short
static const int tickLength = 10;

namespace VideoStitch {
namespace Testing {

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

void testAsyncSeeking() {
  // 10 readers seeking in parallel, make sure it doesn't take 10x seeking time

  int readFrameTime = tickLength * 5;

  MockInput mockInput;
  mockInput.readFrameTime = readFrameTime;

  std::vector<MockInput> inputs{mockInput, mockInput, mockInput, mockInput, mockInput, mockInput};
  auto numReaders = inputs.size();

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  // wait for possible async setup to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(readFrameTime));

  Util::SimpleTimer timer;
  ENSURE(readerController->seekFrame(10), "Seeking should be succesful");

  unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
  if (concurentThreadsSupported >= 4) {
    ENSURE(timer.elapsed() < numReaders * readFrameTime * 1000, "Seeking should happen in parallel");
  }
}

void testAsyncLoading() {
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  std::atomic<int> readFrameCalls(0), readFrameExits(0);

  int readFrameTime = 5 * tickLength;

  FakeReaderFactory* fakeReaderFactory = new FakeReaderFactory(readFrameTime, &readFrameCalls, &readFrameExits);

  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, fakeReaderFactory);
  ENSURE(readerController.status());

  int numReaders = (int)panoDef->numInputs();

  // wait for possible async setup to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(10 * readFrameTime));

  // implementation dependent number of preloaded frames
  int preloadedFrames = readFrameExits;

  ENSURE(preloadedFrames >= numReaders, "Controller init should start pre-loading at least one frame per reader");

  std::map<readerid_t, Input::PotentialFrame> frames;
  std::list<Audio::audioBlockGroupMap_t> audio;
  Input::MetadataChunk metadata;
  mtime_t date;

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);

  ENSURE_EQ((int)readFrameExits, preloadedFrames, "next readFrame should not be finished yet if it's running async");

  std::this_thread::sleep_for(std::chrono::milliseconds(tickLength));
  ENSURE_EQ((int)readFrameCalls, numReaders + preloadedFrames,
            "stitch/extract should trigger load of another frame in the background");

  ENSURE_EQ((int)readFrameCalls, numReaders + preloadedFrames, "no additional frames are pre-loaded");

  // enqueue a couple of loads before destroying controller
  for (int i = 0; i < 5; i++) {
    readerController->load(date, frames, audio, metadata);
    readerController->releaseBuffer(frames);
  }

  // must not crash when leaving this block
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // test timeout 10 seconds
  // if there's a bug in the bufferedReader, it might hang indefinitely
  VideoStitch::Testing::initTestWithTimeoutInSeconds(10);

  auto log = VideoStitch::Logger::get(VideoStitch::Logger::Info);

  log << "Running testAsyncLoading..." << std::endl;
  VideoStitch::Testing::testAsyncLoading();

  log << "Running testAsyncSeeking" << std::endl;
  VideoStitch::Testing::testAsyncSeeking();
  return 0;
}
