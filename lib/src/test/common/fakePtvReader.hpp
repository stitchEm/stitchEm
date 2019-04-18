// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "fakeReader.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"

#include <memory>

namespace VideoStitch {
namespace Testing {

/**
 * A fake reader factory that ignores the given config and creates configurable readers from parameters taken from a
 * customized ptv. Create a list of MockInputs and pass them into `getTestPanoDefWithInputs` to have a high-level mock
 * input reader factory.
 */
class MockPtvReaderFactory : public Input::ReaderFactory {
 public:
  explicit MockPtvReaderFactory(std::atomic<int>* readFrameCalls = nullptr) : readFrameCalls(readFrameCalls) {}

  virtual ~MockPtvReaderFactory() {}

  virtual Potential<Input::Reader> create(readerid_t id, const Core::ReaderInputDefinition& def) const {
    const Ptv::Value& config = def.getReaderConfig();
    FrameRate frameRate{(int)config.has("frameRate.num")->asInt(), (int)config.has("frameRate.den")->asInt()};
    auto readFrameTime = config.has("readFrameTime")->asInt();
    frameid_t firstFrame = (frameid_t)config.has("firstFrame")->asInt();
    auto lastFrame = config.has("lastFrame")->asInt();
    auto isProcedural = config.has("isProcedural")->asBool();
    ReaderClockResolution clockResolution = config.has("videoClockInMicroSeconds")->asBool()
                                                ? ReaderClockResolution::Microsecond
                                                : ReaderClockResolution::Millisecond;
    bool hasVideo = def.getIsVideoEnabled();
    bool hasAudio = def.getIsAudioEnabled();
    AudioMockTest testCase = AudioMockTest::NormalInput;

    if (testSettings.find(id) != testSettings.end()) {
      testCase = testSettings.at(id);
    }

    if (hasVideo && hasAudio) {
      // Audio Video reader
      return new FakeAVReader(id, def.getWidth(), def.getHeight(), frameRate,
                              (frameid_t)firstFrame + def.getFrameOffset(), (frameid_t)lastFrame + def.getFrameOffset(),
                              nullptr, (int)readFrameTime, isProcedural, testCase, readFrameCalls);
    } else if (!hasVideo && hasAudio) {
      // Audio only
      return new FakeAudioReader(id, frameRate, firstFrame, testCase);
    } else if (hasVideo && !hasAudio) {
      // Video only
      return new FakeVideoReader(
          id, def.getWidth(), def.getHeight(), frameRate, clockResolution, (frameid_t)firstFrame + def.getFrameOffset(),
          (frameid_t)lastFrame + def.getFrameOffset(), nullptr, (int)readFrameTime, isProcedural, readFrameCalls);
    }
    std::cout << "Reader without audio and video -> not possible" << std::endl;
    return nullptr;
  }

  virtual Input::ProbeResult probe(const Ptv::Value& /*config*/) const {
    ENSURE(false, "not supported");
    return Input::ProbeResult({false, false, -1, -1, -1, -1});
  }

  virtual int getFirstFrame() const { return 0; }

  virtual int getNumFrames() const { return -1; }

  void setTestSetting(int id, AudioMockTest testCase) { testSettings[id] = testCase; }

 private:
  std::map<int, AudioMockTest> testSettings;
  std::atomic<int>* readFrameCalls;
};

class MockInput {
 public:
  FrameRate frameRate = {25, 1};
  ReaderClockResolution videoClock{ReaderClockResolution::Microsecond};
  int firstFrame = 0;
  int lastFrame = -1;
  int readFrameTime = 0;
  bool readVideo = true;
  bool readAudio = false;
  int frameOffset = 0;
  bool isProcedural = false;
  int readerGroupID = -1;
};

inline Core::PanoDefinition* getTestPanoDefWithInputs(std::vector<MockInput> inputs) {
  Potential<Ptv::Parser> parser(Ptv::Parser::create());

  std::stringstream ss;
  ss << "{"
        " \"width\": 513, "
        " \"height\": 315, "
        " \"hfov\": 90.0, "
        " \"proj\": \"rectilinear\", "
        " \"inputs\": [ ";

  bool firstInput = true;
  for (auto input : inputs) {
    std::stringstream groupID;
    if (input.readerGroupID >= 0) {
      groupID << "   \"group\": " << input.readerGroupID << ", ";
    }

    if (!firstInput) {
      ss << ",";
    }
    ss << "  { "
          "   \"width\": 17, "
          "   \"height\": 13, "
          "   \"hfov\": 90.0, "
          "   \"yaw\": 0.0, "
          "   \"pitch\": 0.0, "
          "   \"roll\": 0.0, "
          "   \"proj\": \"rectilinear\", "
          "   \"viewpoint_model\": \"ptgui\", "
          "   \"response\": \"linear\", "
       << groupID.str() << "   \"frame_offset\": " << input.frameOffset << " , "
       << "   \"audio_enabled\": " << (input.readAudio ? "true" : "false") << ","
       << "   \"video_enabled\": " << (input.readVideo ? "true" : "false") << ","
       << "   \"reader_config\": { "
       << "   \"frameRate.num\": " << input.frameRate.num
       << ","
          "   \"frameRate.den\": "
       << input.frameRate.den
       << ","
          "   \"videoClockInMicroSeconds\": "
       << (input.videoClock == ReaderClockResolution::Microsecond ? true : false)
       << ","
          "   \"firstFrame\": "
       << input.firstFrame
       << ","
          "   \"lastFrame\": "
       << input.lastFrame
       << ","
          "   \"readFrameTime\": "
       << input.readFrameTime
       << ","
          "   \"isProcedural\": "
       << input.isProcedural
       << "}"
          "  } ";
    firstInput = false;
  }
  ss << " ]"
        "}";

  if (!parser->parseData(ss.str())) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return NULL;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);
  return panoDef.release();
}

}  // namespace Testing
}  // namespace VideoStitch
