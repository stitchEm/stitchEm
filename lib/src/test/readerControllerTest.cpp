// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "common/fakeReader.hpp"
#include "common/fakePtvReader.hpp"
#include "audio/orah/orahAudioSync.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/ptv.hpp"
#include <core/readerController.hpp>

#include <memory>

namespace VideoStitch {
namespace Testing {

// how many milliseconds per time-sensitive testing step?
// increase (e.g. to 100) if test fails due to being too short
static const int tickLength = 1;

Core::PanoDefinition *getTestPanoDef() {
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

void testReaderSpec() {
  std::vector<MockInput> inputs;
  for (int i = 1; i < 11; i++) {
    MockInput input;
    input.frameRate = {2 * i, 3 * i};
    input.firstFrame = 4 * i;
    input.lastFrame = 5 * i * 1000;
    input.readFrameTime = 6 * i;
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  ENSURE(readerController->getPano().numInputs() == 10);

  for (int inputID = 1; inputID < 11; ++inputID) {
    auto spec = readerController->getReaderSpec(inputID - 1);
    ENSURE_EQ(FrameRate{2 * inputID, 3 * inputID}, spec.frameRate,
              "frameRate of MockReader should be reported in Reader::Spec");
    ENSURE_EQ(5 * inputID * 1000 - 4 * inputID + 1, spec.frameNum,
              "number of frames of MockReader should be reported in Reader::Spec");
  }

  ENSURE_EQ(40, readerController->getFirstReadableFrame(),
            "first readable frame reported by reader controller should be first frame of the latest starting reader");
  ENSURE_EQ(5000, readerController->getLastReadableFrame(),
            "last readable frame reported by reader controller should be last frame of the first ending reader");
}

void testLoadedDate() {
  // readers with different load dates, test current implementation
  std::vector<MockInput> inputs;
  // reader 1-based ID == 5 and ID == 4 are audio readers
  for (int i = 1; i < 11; i++) {
    MockInput input;
    input.frameRate = {1, 1};
    input.firstFrame = i < 7 ? 5 : 0;
    input.readAudio = (i == 4 || i == 5);
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  Input::MetadataChunk metadata;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;

  readerController->load(date, frames, audio, metadata);
  // Audio reader is reader #5, its first frame is #5, at 1 fps
  ENSURE_EQ(date, (mtime_t)5 * 1000 * 1000, "ReaderController should take timing information from audio reader");
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");
  for (auto &audioGr : audio) {
    for (auto &blockmap : audioGr) {
      ENSURE_EQ((int)blockmap.second.size(), 2, "Should return a block with two inputs");
      ENSURE_EQ((int)blockmap.second.begin()->first, 3, "Check first input id audio");
      ENSURE_EQ((int)blockmap.second.rbegin()->first, 4, "Check second input id audio");
    }
  }

  readerController->releaseBuffer(frames);
  audio.clear();

  ENSURE_EQ(5, readerController->getCurrentFrame(), "Read only one frame, audio reader starts at frame 5");

  readerController->load(date, frames, audio, metadata);
  ENSURE_EQ(date, (mtime_t)6 * 1000 * 1000, "ReaderController should take timing information from audio reader");
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");
  readerController->releaseBuffer(frames);
  audio.clear();

  ENSURE_EQ(6, readerController->getCurrentFrame(), "Read two frames");

  readerController->load(date, frames, audio, metadata);
  ENSURE_EQ(date, (mtime_t)7 * 1000 * 1000, "ReaderController should take timing information from audio reader");
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");
  readerController->releaseBuffer(frames);
  audio.clear();

  ENSURE_EQ(7, readerController->getCurrentFrame(), "Read three frames");

  date = readerController->reload(frames);
  ENSURE_EQ(date, (mtime_t)7 * 1000 * 1000, "ReaderController should return same time on reloading");
  readerController->releaseBuffer(frames);

  ENSURE_EQ(7, readerController->getCurrentFrame(), "Current frame shouldn't change, only reloaded");

  readerController->load(date, frames, audio, metadata);
  ENSURE_EQ(date, (mtime_t)8 * 1000 * 1000, "ReaderController should return next frame after reloading");
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");
  readerController->releaseBuffer(frames);
  audio.clear();

  ENSURE_EQ(8, readerController->getCurrentFrame(), "Read four frames");

  ENSURE(readerController->seekFrame(0));
  readerController->load(date, frames, audio, metadata);
  ENSURE_EQ(date, (mtime_t)0 * 1000 * 1000, "ReaderController should return next frame after reloading");
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");
  readerController->releaseBuffer(frames);
  audio.clear();
}

void testLoadedDateWithAudioReader() {
  const int firstVideoFrameToTest = 123;

  // one audio-only reader, one video/audio reader
  std::vector<MockInput> inputs;
  for (int i = 0; i < 2; i++) {
    MockInput input;
    input.frameRate = {1, 1};
    input.firstFrame = firstVideoFrameToTest * i;
    input.readVideo = (i == 1);
    input.readAudio = true;
    input.readerGroupID = i;
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  Input::MetadataChunk metadata;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;

  readerController->load(date, frames, audio, metadata);
  ENSURE_EQ(date, (mtime_t)firstVideoFrameToTest * 1000 * 1000,
            "ReaderController should take timing information from audio/video reader");
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");
  readerController->releaseBuffer(frames);
  audio.clear();

  ENSURE_EQ(123, readerController->getCurrentFrame(),
            "Read only one frame, audio/video reader starts at frame `firstVideoFrameToTest`");
}

void testSeekWithImage() {
  // Check that the image reader sync is ignored
  // and that seek load takes less then twice the time
  // of a single video reader
  std::vector<MockInput> inputs;
  for (int i = 0; i < 2; ++i) {
    MockInput input;
    if (i == 0) {
      input.frameRate = {100, 1};
    } else {
      input.frameRate = {60, 1};
    }
    input.firstFrame = 0;
    input.isProcedural = (i != 0);
    input.readAudio = true;
    input.readerGroupID = 0;
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  std::atomic<int> readFrameCalls(0);
  auto readerFactory = new MockPtvReaderFactory(&readFrameCalls);
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  Input::MetadataChunk metadata;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);

  frameid_t frameId = 200;
  ENSURE(readerController->seekFrame(frameId), "Seeking should be succesful");

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  // Reader #1 should be ignored, reader #0 date should be used
  mtime_t expectedDate = (mtime_t)frameId * 1000 * 1000 / inputs[0].frameRate.num;
  ENSURE_EQ(expectedDate, date, "ReaderController should take timing information from the other reader");

  frameId = 350;
  ENSURE(readerController->seekFrame(frameId), "Seeking should be succesful");
  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  expectedDate = (mtime_t)frameId * 1000 * 1000 / inputs[0].frameRate.num;
  ENSURE_EQ(expectedDate, date, "ReaderController should take timing information from the other reader");

  frameId = 500;
  ENSURE(readerController->seekFrame(frameId), "Seeking should be succesful");
  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  delete readerController.release();
  expectedDate = (mtime_t)frameId * 1000 * 1000 / inputs[0].frameRate.num;
  ENSURE_EQ(expectedDate, date, "ReaderController should take timing information from the other reader");
  // Seek and load should not drop any frame
  ENSURE(readFrameCalls.load() < 100, "Procedural inputs should not be synched");
}

void testEOSAudio() {
  // readers with different load dates, test current implementation
  std::vector<MockInput> inputs;
  // reader ID == 0 and ID == 1 are audio readers
  for (int i = 0; i < 3; i++) {
    MockInput input;
    input.frameRate = {1, 1};
    input.readAudio = (i == 0 || i == 1);
    inputs.push_back(input);
  }
  auto readerFactory = new MockPtvReaderFactory();
  // Test EOS
  readerFactory->setTestSetting(1, AudioMockTest::EOSInput);

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  Input::MetadataChunk metadata;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;

  std::tuple<Input::ReadStatus, Input::ReadStatus, Input::ReadStatus> status =
      readerController->load(date, frames, audio, metadata);
  Input::ReadStatus audioReadStatus = std::get<1>(status);
  ENSURE_EQ((int)Input::ReadStatusCode::EndOfFile, (int)audioReadStatus.getCode(), "Check audio read status");
  ENSURE_EQ(audio.empty(), true, "audio should be empty");
  readerController->releaseBuffer(frames);
}

void testTryAgainAudio() {
  // readers with different load dates, test current implementation
  std::vector<MockInput> inputs;
  // reader ID == 0 and ID == 1 are audio readers
  for (int i = 0; i < 3; i++) {
    MockInput input;
    input.frameRate = {1, 1};
    input.readAudio = (i == 0 || i == 1);
    inputs.push_back(input);
  }
  auto readerFactory = new MockPtvReaderFactory();
  // Test EOS
  readerFactory->setTestSetting(1, AudioMockTest::TryAgainInput);

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  Input::MetadataChunk metadata;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;

  std::tuple<Input::ReadStatus, Input::ReadStatus, Input::ReadStatus> status =
      readerController->load(date, frames, audio, metadata);
  Input::ReadStatus audioReadStatus = std::get<1>(status);
  ENSURE_EQ((int)Input::ReadStatusCode::TryAgain, (int)audioReadStatus.getCode(), "Check audio read status");
  ENSURE_EQ(audio.empty(), true, "audio should be empty");
  readerController->releaseBuffer(frames);
}

void testAudioVideoResync() {
  // readers with different load dates, test current implementation
  std::vector<MockInput> inputs;
  // reader ID == 0 and ID == 1 are audio readers
  for (int i = 0; i < 3; i++) {
    MockInput input;
    input.frameRate = {1, 1};
    input.readVideo = (i == 2);
    input.readAudio = (i == 0 || i == 1);
    inputs.push_back(input);
  }
  auto readerFactory = new MockPtvReaderFactory();

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  Input::MetadataChunk metadata;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;

  std::tuple<Input::ReadStatus, Input::ReadStatus, Input::ReadStatus> status =
      readerController->load(date, frames, audio, metadata);
  ENSURE_EQ(audio.empty(), false, "audio should not be empty");

  int iBlk = 0;

  int blockSize = Audio::getDefaultBlockSize();
  auto sr = Audio::getDefaultSamplingRate();
  mtime_t blockDuration = (mtime_t)std::round((blockSize * 1000000. / sr));
  for (auto &grMap : audio) {
    for (auto &readerMap : grMap) {
      for (auto &kv : readerMap.second) {
        mtime_t tmp = (blockDuration * iBlk);
        ENSURE_EQ((mtime_t)(date + tmp), kv.second.getTimestamp(), "test audio video synchro");
      }
    }
    iBlk++;
  }
  ENSURE_EQ(date, audio.front()[0][0].getTimestamp(), "audio and video should be synchro");
  readerController->releaseBuffer(frames);
}

void testInputGroups(int fps, ReaderClockResolution clockResolution) {
  // grouped readers with date on the first frame should be synchronized
  double frameTime = 1000. / fps;  // milliseconds

  std::vector<MockInput> inputs;
  // grouped readers
  for (int i = 1; i < 11; i++) {
    MockInput input;
    input.frameRate = {fps, 1};
    input.videoClock = clockResolution;
    // first frames are reader index
    input.firstFrame = i;
    // all readers are part of a group
    input.readerGroupID = 0;
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  auto checkFrameID = [&](int expectedFrameID) {
    mtime_t expectedFrameTime;

    switch (clockResolution) {
      case ReaderClockResolution::Microsecond:
        expectedFrameTime = (mtime_t)(1000. * (double)(expectedFrameID * frameTime));
        break;
      case ReaderClockResolution::Millisecond:
        expectedFrameTime = 1000 * (mtime_t)(expectedFrameID * frameTime);
        break;
      default:
        assert(false);
        break;
    }

    std::map<readerid_t, Input::PotentialFrame> frames;
    std::list<Audio::audioBlockGroupMap_t> audio;
    mtime_t date;
    Input::MetadataChunk metadata;

    readerController->load(date, frames, audio, metadata);
    ENSURE_EQ(expectedFrameTime, date, "All readers are grouped, should have forwarded to latest frame ID");
    ENSURE_EQ(expectedFrameID, readerController->getCurrentFrame(),
              "All readers are grouped, should have forwarded to latest frame ID");

    for (int readerID = 0; readerID < 10; readerID++) {
      ENSURE(frames[readerID].status.ok(), "All frames should have loaded correctly");
      mtime_t *videoFrame = (mtime_t *)frames[readerID].frame.hostBuffer().hostPtr();
      ENSURE_EQ(expectedFrameTime, *videoFrame,
                "Mock Reader sets its time stamp as the videoFrame content. It should be the shared for all readers in "
                "the group.");
    }
    readerController->releaseBuffer(frames);

    date = readerController->reload(frames);
    ENSURE_EQ(expectedFrameTime, date, "All readers are grouped, should have forwarded to latest frame ID");
    ENSURE_EQ(expectedFrameID, readerController->getCurrentFrame(),
              "All readers are grouped, should have forwarded to latest frame ID");

    for (int readerID = 0; readerID < 10; readerID++) {
      ENSURE(frames[readerID].status.ok(), "All frames should have loaded correctly");
      mtime_t *videoFrame = (mtime_t *)frames[readerID].frame.hostBuffer().hostPtr();
      ENSURE_EQ(expectedFrameTime, *videoFrame,
                "Mock Reader sets its time stamp as the videoFrame content. It should be the shared for all readers in "
                "the group.");
    }
    readerController->releaseBuffer(frames);
  };

  int firstFrameID = 10;  // last reader is latest reader
  checkFrameID(firstFrameID);
  checkFrameID(firstFrameID + 1);
  checkFrameID(firstFrameID + 2);
}

Potential<Core::ReaderController> createReaderControllerWithInputFrameRates(
    std::vector<std::pair<FrameRate, bool>> inputFrameRates) {
  std::vector<MockInput> inputs;

  for (std::pair<FrameRate, bool> inputInfo : inputFrameRates) {
    MockInput input;
    input.frameRate = inputInfo.first;
    input.isProcedural = inputInfo.second;
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  return Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
}

void testReaderFrameRate() {
  {
    auto readerController =
        createReaderControllerWithInputFrameRates({{{127, 131}, false}, {{127, 131}, false}, {{127, 131}, false}});
    ENSURE(readerController.status(), "ReaderController should have set up ok");
    ENSURE_EQ(FrameRate{127, 131}, readerController->getFrameRate(),
              "ReaderController should report frame rate that is shared by all readers");
  }

  {
    // not equal numbers, but equal frame rate
    auto readerController =
        createReaderControllerWithInputFrameRates({{{1, 1}, false}, {{2, 2}, false}, {{3, 3}, false}});
    ENSURE(readerController.status(), "ReaderController should have set up ok ");
    ENSURE_EQ(FrameRate{1, 1}, readerController->getFrameRate(),
              "ReaderController should report frame rate that is shared by all readers");
  }

  {
    auto readerController = createReaderControllerWithInputFrameRates({{{1, 1}, false}, {{1, 2}, false}});
    ENSURE(readerController.status().getType() == ErrType::InvalidConfiguration,
           "ReaderController should fail setup with different video frame rates");
  }

  {
    // video + procedural: --> pick video
    auto readerController =
        createReaderControllerWithInputFrameRates({{{1, 2}, false}, {{1, 2}, false}, {{1, 3}, true}});
    ENSURE(readerController.status(), "ReaderController should have set up ok ");
    ENSURE_EQ(FrameRate{1, 2}, readerController->getFrameRate(),
              "ReaderController should report frame rate from the video (non-procedural) readers");
  }

  {
    // procedural only, but differnt --> don't care, get me some friggn framerate
    auto readerController = createReaderControllerWithInputFrameRates({{{1, 3}, true}, {{1, 4}, true}, {{1, 5}, true}});
    ENSURE(readerController.status(), "ReaderController should have set up ok ");
    ENSURE_EQ(FrameRate{VIDEO_WRITER_DEFAULT_FRAMERATE_NUM, VIDEO_WRITER_DEFAULT_FRAMERATE_DEN},
              readerController->getFrameRate(),
              "ReaderController should fall back to default frame rate with differing procedural readers");
  }
}

void testCurrentFrame() {
  std::vector<MockInput> inputs;
  for (int i = 1; i < 11; i++) {
    // readers w/ different frameOffsets
    MockInput input;
    input.frameOffset = i * 7;
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  auto readerFactory = new MockPtvReaderFactory();
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());

  std::map<readerid_t, Input::PotentialFrame> frames;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t date;
  Input::MetadataChunk metadata;

  // load / getCurrentFrame
  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(0, readerController->getCurrentFrame(), "First frame");

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(1, readerController->getCurrentFrame(), "Second frame");

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(2, readerController->getCurrentFrame(), "Third frame");

  // reload
  readerController->reload(frames);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(2, readerController->getCurrentFrame(), "Reloaded third frame");

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(3, readerController->getCurrentFrame(), "Fourth frame");

  // seek
  readerController->seekFrame(50);
  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(50, readerController->getCurrentFrame(), "Seek to frame 50");

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(51, readerController->getCurrentFrame(), "Frame 51");

  // reload
  readerController->reload(frames);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(51, readerController->getCurrentFrame(), "Reload frame 51");

  // seek backwards
  readerController->seekFrame(7);
  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(7, readerController->getCurrentFrame(), "Seek to frame 5");

  readerController->load(date, frames, audio, metadata);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(8, readerController->getCurrentFrame(), "Frame 8");

  // reload
  readerController->reload(frames);
  readerController->releaseBuffer(frames);
  ENSURE_EQ(8, readerController->getCurrentFrame(), "Reload frame 8");
}

void testReaderLifeTime() {
  std::vector<std::atomic<int> *> readerChecks;
  std::atomic<int> factoryCheck(0);

  {
    std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
    std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());

    auto factory = new ResourceCheckReaderFactory(readerChecks, factoryCheck);
    auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, factory);
    ENSURE(readerController.status());

    ENSURE(factoryCheck == 1, "Reader factory should have been deleted after ReaderController creation");

    ENSURE_EQ((size_t)2, readerChecks.size(), "Two readers should have been created from the test pano Def");
  }

  ENSURE_EQ(1, readerChecks[0]->load(), "Reader 0 should have been deallocated once");
  ENSURE_EQ(1, readerChecks[1]->load(), "Reader 1 should have been deallocated once");

  for (auto ptr : readerChecks) {
    delete ptr;
  }
}

void testAudioVideoSynchro() {
  // Test audio video synchronization with a stitching box configuration like
  // that is to say 2 different groups of readers:
  // - group 0: 2 audio/video readers
  // - group 1: 1 audio reader

  // grouped readers with date on the first frame should be synchronized
  int fps = 50;
  int audioVideoGrId = 44;
  int audioOnlyGrId = 55;

  std::vector<MockInput> inputs;
  // grouped readers
  for (int i = 0; i < 3; i++) {
    MockInput input;
    input.frameRate = {fps, 1};
    // all readers are part of a group
    if (i < 2) {
      // First two readers are in the same group 0
      if (i == 0) {
        input.firstFrame = 1;
      } else {
        input.firstFrame = 5;
      }
      input.readerGroupID = audioVideoGrId;
      input.readAudio = true;
      input.readVideo = true;
    } else {
      input.firstFrame = 10;
      input.readerGroupID = audioOnlyGrId;
      input.readAudio = true;
      input.readVideo = false;
    }
    inputs.push_back(input);
  }

  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);
  ENSURE(readerController.status());
  std::map<readerid_t, Input::PotentialFrame> frames;
  std::list<Audio::audioBlockGroupMap_t> audio;
  mtime_t videoDate;
  Input::MetadataChunk metadata;

  // Let's load 100 frames and check if the syncrhonization remains stable
  for (int iTurn = 0; iTurn < 100; iTurn++) {
    readerController->load(videoDate, frames, audio, metadata);
    readerController->releaseBuffer(frames);
    if (iTurn == 0) {
      size_t nbAudioBlocksExpected =
          (size_t)((double)(Core::kAudioPreRoll) / (1000. * 1000.) /
                       ((double)(audioPipe->getBlockSize()) / double(audioPipe->getSamplingRate())) +
                   1.5);
      ENSURE_EQ(nbAudioBlocksExpected, audio.size(), "Check number of audio blocks at the first load");
      // Check that the first audio frame of the audio video group are well synchronized
      ENSURE_EQ(videoDate, audio.front().at(audioVideoGrId).at(0).getTimestamp(),
                "Check audio video synchronization of the audio video group");
      ENSURE_EQ(videoDate, audio.front().at(audioVideoGrId).at(1).getTimestamp(),
                "Check audio video synchronization of the audio video group");
      ENSURE_EQ(videoDate, audio.front().at(audioOnlyGrId).at(2).getTimestamp(),
                "Check audio video synchronization of the audio video group");
    }

    for (const Audio::audioBlockGroupMap_t &audioPerGroup : audio) {
      ENSURE_EQ((size_t)2, audioPerGroup.size(), "Check number of groups loaded");
      mtime_t timestampAudioVideo = -1;
      mtime_t timestampAudioOnly = -1;
      for (auto &kvGroup : audioPerGroup) {
        groupid_t grId = kvGroup.first;
        const Audio::audioBlockReaderMap_t &audioPerReader = kvGroup.second;
        if (grId == audioVideoGrId) {
          ENSURE_EQ(audioPerReader.at(0).getTimestamp(), audioPerReader.at(1).getTimestamp(),
                    "Check timestamps of audio video group are aligned");
          timestampAudioVideo = audioPerReader.at(0).getTimestamp();
        } else {
          timestampAudioOnly = audioPerReader.at(2).getTimestamp();
          ENSURE(timestampAudioVideo > -1 && timestampAudioOnly > -1 &&
                 timestampAudioVideo == timestampAudioOnly);  // Check timestamps are still synchronized
        }
      }
    }
    // Check timestamps of the first audio blocks
    audio.clear();
  }
}

typedef Audio::Orah::orahSample_t sample_t;

// Create blanks.
// Mark one sample before and one after to check for correct blanking
static void genData(sample_t *buf, int len, int offset) {
  memset(buf, 4, sizeof(sample_t) * len);
  for (int i = 0, k = 0; i < ORAH_SYNC_NUM_BLANKS; i++) {
    if (offset > 0 || i != 0) {
      buf[i * 88 * 2 + k - 1 + offset] = 12345;
    }
    for (int j = 0; j < 88; j++, k++) {
      buf[i * 88 * 2 + j + offset] = 0;
    }
    buf[i * 88 * 2 + k + offset] = 12345;
  }
}

void testAudioPreProc() {
  int readFrameTime = tickLength;
  MockInput mockInput;
  mockInput.readFrameTime = readFrameTime;

  std::vector<MockInput> inputs{mockInput, mockInput, mockInput, mockInput, mockInput, mockInput};
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDefWithInputs(inputs));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto readerFactory = new MockPtvReaderFactory();
  auto readerController = Core::ReaderController::create(*panoDef, *audioPipe, readerFactory);

  groupid_t gr = 0;
  readerController->setupAudioPreProc(Audio::Orah::getOrahAudioSyncName(), gr);

  std::map<readerid_t, Audio::Samples> inOutMap;
  int bPos{151}, blockSize{1024} /* 512 samples * 2 (interleaved) channels*/, fakeDataSize{blockSize * 20}, n{0};
  // Generated data
  std::vector<sample_t> data[2];  // Two-channel "streams"
  data[0].resize(fakeDataSize);
  data[1].resize(fakeDataSize);

  genData(data[0].data(), fakeDataSize, 0);
  genData(data[1].data(), fakeDataSize, bPos);
  auto ski1 = std::search_n(data[0].begin(), data[0].end(), 5, 0);
  auto ski2 = std::search_n(data[1].begin(), data[1].end(), 5, 0);
  ENSURE(ski1 != data[0].end(), "Blank 1 should be present");
  ENSURE(ski2 != data[1].end(), "Blank 2 should be present");

  while (n < fakeDataSize) {
    // Input
    Audio::Samples::data_buffer_t block1;
    Audio::Samples::data_buffer_t block2;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    std::copy(&data[0][n], &data[0][n] + blockSize, (sample_t *)block1[0]);
    std::copy(&data[1][n], &data[1][n] + blockSize, (sample_t *)block2[0]);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    inOutMap[0] = std::move(s2);
    inOutMap[1] = std::move(s1);
    readerController->applyAudioPreProc(inOutMap, gr);
    n += blockSize;
    ENSURE((int)inOutMap.size() == 2, "Check out map size.");
    ENSURE(inOutMap.find(0) != inOutMap.end(), "Check out map reader id.");
    ENSURE(inOutMap.find(1) != inOutMap.end(), "Check out map reader id.");
    ENSURE_EQ(audioPipe->getBlockSize(), (int)inOutMap.find(0)->second.getNbOfSamples(), "Check out block size.");
    ENSURE_EQ(audioPipe->getBlockSize(), (int)inOutMap.find(1)->second.getNbOfSamples(), "Check out block size.");
    std::vector<sample_t> a1(&((sample_t **)inOutMap[0].getSamples().data())[0][0],
                             &((sample_t **)inOutMap[0].getSamples().data())[0][blockSize]);
    std::vector<sample_t> a2(&((sample_t **)inOutMap[1].getSamples().data())[0][0],
                             &((sample_t **)inOutMap[1].getSamples().data())[0][blockSize]);
    auto sk1 = std::search_n(a1.begin(), a1.end(), 5, 0);
    auto sk2 = std::search_n(a2.begin(), a2.end(), 5, 0);
    ENSURE(sk1 == a1.end(), "Blank 1 not erased");
    ENSURE(sk2 == a2.end(), "Blank 2 not erased");
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // test timeout 10 seconds
  // if there's a bug in the bufferedReader, it might hang indefinitely
  VideoStitch::Testing::initTestWithTimeoutInSeconds(10);

  auto log = VideoStitch::Logger::get(VideoStitch::Logger::Info);

  log << "Running testReaderSpec" << std::endl;
  VideoStitch::Testing::testReaderSpec();

  log << "Running testLoadedDate" << std::endl;
  VideoStitch::Testing::testLoadedDate();

  log << "Running testInputGroups" << std::endl;
  VideoStitch::Testing::testInputGroups(30, VideoStitch::Testing::ReaderClockResolution::Microsecond);
  VideoStitch::Testing::testInputGroups(50, VideoStitch::Testing::ReaderClockResolution::Microsecond);

  // VSA-5761
  log << "Running testInputGroups" << std::endl;
  VideoStitch::Testing::testInputGroups(30, VideoStitch::Testing::ReaderClockResolution::Millisecond);
  VideoStitch::Testing::testInputGroups(50, VideoStitch::Testing::ReaderClockResolution::Millisecond);

  log << "Running testReaderFrameRate" << std::endl;
  VideoStitch::Testing::testReaderFrameRate();

  log << "Running testCurrentFrame" << std::endl;
  VideoStitch::Testing::testCurrentFrame();

  log << "Running testReaderLifeTime" << std::endl;
  VideoStitch::Testing::testReaderLifeTime();

  log << "Running testEOSAudio" << std::endl;
  VideoStitch::Testing::testEOSAudio();

  log << "Running testTryAgainAudio" << std::endl;
  VideoStitch::Testing::testTryAgainAudio();

  log << "Running testAudioVideoResync" << std::endl;
  VideoStitch::Testing::testAudioVideoResync();

  log << "Running testAudioVideoSynchro" << std::endl;
  VideoStitch::Testing::testAudioVideoSynchro();

  log << "Running testAudioPreProc" << std::endl;
  VideoStitch::Testing::testAudioPreProc();

  log << "Running testSeekWithImage" << std::endl;
  VideoStitch::Testing::testSeekWithImage();

  return 0;
}
