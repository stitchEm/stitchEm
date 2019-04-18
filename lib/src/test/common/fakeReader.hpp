// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "libvideostitch/input.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"

#include "../gpu/testing.hpp"

#include <atomic>
#include <thread>

namespace VideoStitch {
namespace Testing {

enum class AudioMockTest { EOSInput = 0, TryAgainInput, NormalInput };

enum class ReaderClockResolution {
  Millisecond,
  Microsecond,
};

/**
 * A fake reader that
 * - sleeps for readFrameTime when reading frames and seeking
 * - tracks number of calls to readFrame (readFrameCalls, readFrameExits)
 * - simulates a date computed through frame rate and given first frame
 * - stores the date in the first 8 Bytes of the frame
 */
class FakeVideoReader : public Input::VideoReader {
 public:
  FakeVideoReader(readerid_t id, int64_t width, int64_t height, FrameRate frameRate,
                  ReaderClockResolution clockResolution, frameid_t firstFrame, frameid_t lastFrame,
                  const unsigned char *maskHostBuffer, int readFrameTime, bool isProcedural,
                  std::atomic<int> *readFrameCalls = nullptr, std::atomic<int> *readFrameExits = nullptr)
      : Reader(id),
        VideoReader(width, height, width * height * 4, PixelFormat::RGBA, Host, frameRate, firstFrame, lastFrame,
                    isProcedural, maskHostBuffer),
        frameRate(frameRate),
        clockResolution(clockResolution),
        frameID(firstFrame),
        readFrameCalls(readFrameCalls),
        readFrameExits(readFrameExits),
        readFrameTime(readFrameTime) {
    // storing the date in the video buffer on readFrame
    assert(width * height * 4 >= (int64_t)sizeof(mtime_t));
  }
  virtual ~FakeVideoReader() {}

  virtual Input::ReadStatus readFrame(mtime_t &date, unsigned char *dst) {
    if (readFrameCalls != nullptr) {
      (*readFrameCalls)++;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(readFrameTime));
    switch (clockResolution) {
      case ReaderClockResolution::Microsecond:
        date = frameID * 1000000 * frameRate.den / frameRate.num;
        break;
      case ReaderClockResolution::Millisecond:
        date = frameID * 1000 * frameRate.den / frameRate.num;
        date *= 1000;
        break;
      default:
        assert(false);
        break;
    }
    mtime_t *dstlong = (mtime_t *)dst;
    *dstlong = date;
    ++frameID;
    if (readFrameExits != nullptr) {
      (*readFrameExits)++;
    }
    return Input::ReadStatus::OK();
  }

  virtual Status seekFrame(frameid_t targetFrame) {
    std::this_thread::sleep_for(std::chrono::milliseconds(readFrameTime));
    frameID = targetFrame;
    return Status::OK();
  }

 private:
  FrameRate frameRate;
  ReaderClockResolution clockResolution;
  frameid_t frameID;
  std::atomic<int> *readFrameCalls;
  std::atomic<int> *readFrameExits;
  int readFrameTime;
};

class FakeAudioReader : public Input::AudioReader {
 public:
  FakeAudioReader(readerid_t id, FrameRate frameRate, frameid_t firstFrame, AudioMockTest testCase)
      : Reader(id),
        AudioReader(Audio::ChannelLayout::STEREO, Audio::SamplingRate::SR_44100, Audio::SamplingDepth::DBL_P),
        timestamp(0),
        testCase(testCase) {
    timestamp = static_cast<mtime_t>(firstFrame) * static_cast<mtime_t>(frameRate.den) * 1000000 /
                static_cast<mtime_t>(frameRate.num);
  }
  virtual ~FakeAudioReader() {}

  virtual Input::ReadStatus readSamples(size_t nbSamples, Audio::Samples &audioSamples) {
    Audio::audioSample_t *data[MAX_AUDIO_CHANNELS];
    // Manage only format stereo planar double
    data[Audio::getChannelIndexFromChannelMap(Audio::SPEAKER_FRONT_LEFT)] = new Audio::audioSample_t[nbSamples];
    data[Audio::getChannelIndexFromChannelMap(Audio::SPEAKER_FRONT_RIGHT)] = new Audio::audioSample_t[nbSamples];

    audioSamples = Audio::Samples(getSpec().sampleRate, getSpec().sampleDepth, getSpec().layout,
                                  static_cast<mtime_t>(timestamp), (uint8_t **)data, nbSamples);
    mtime_t a =
        static_cast<mtime_t>(std::round(nbSamples * 1000000. / Audio::getIntFromSamplingRate(getSpec().sampleRate)));
    timestamp += a;
    return Input::ReadStatus::OK();
  }

  virtual Status seekFrame(mtime_t date) { return Status::OK(); }

  virtual size_t available() {
    if (testCase == AudioMockTest::TryAgainInput || testCase == AudioMockTest::EOSInput) {
      return 0;
    }
    return 1024;
  }

  virtual bool eos() {
    if (testCase == AudioMockTest::EOSInput) {
      return true;
    }
    return false;
  }

 private:
  mtime_t timestamp;
  AudioMockTest testCase;
};

class FakeAVReader : public FakeVideoReader, public FakeAudioReader {
 public:
  FakeAVReader(readerid_t id, int64_t width, int64_t height, FrameRate frameRate, frameid_t firstFrame,
               frameid_t lastFrame, const unsigned char *maskHostBuffer, int readFrameTime, bool isProcedural,
               AudioMockTest testCase, std::atomic<int> *readFrameCalls = nullptr,
               std::atomic<int> *readFrameExits = nullptr,
               ReaderClockResolution clockResolution = ReaderClockResolution::Microsecond)
      : Reader(id),
        FakeVideoReader(id, width, height, frameRate, clockResolution, firstFrame, lastFrame, maskHostBuffer,
                        readFrameTime, isProcedural, readFrameCalls, readFrameExits),
        FakeAudioReader(id, frameRate, firstFrame, testCase) {}

  virtual ~FakeAVReader() {}
};

/**
 * A fake reader factory that ignores the given config and creates configurable readers
 * from arguments at factory creation time
 */
class FakeReaderFactory : public Input::ReaderFactory {
 public:
  FakeReaderFactory(int readFrameTime, std::atomic<int> *readFrameCalls = nullptr,
                    std::atomic<int> *readFrameExits = nullptr)
      : frameRate({10, 0}),
        firstFrame(0),
        lastFrame(10),
        maskHostBuffer(NULL),
        readFrameCalls(readFrameCalls),
        readFrameExits(readFrameExits),
        readFrameTime(readFrameTime) {}

  /**
   * Reader option setters.
   * @{
   */
  void setFrameRate(FrameRate value) { frameRate = value; }
  void setFirstFrame(frameid_t value) { firstFrame = value; }
  void setLastFrame(frameid_t value) { lastFrame = value; }
  void setMaskHostBuffer(const unsigned char *value) { maskHostBuffer = value; }
  /**
   * @}
   */

  virtual ~FakeReaderFactory() {}

  virtual Potential<Input::Reader> create(readerid_t id, const Core::ReaderInputDefinition &def) const {
    return Potential<Input::Reader>(
        new FakeVideoReader(id, def.getWidth(), def.getHeight(), frameRate, ReaderClockResolution::Microsecond,
                            firstFrame + def.getFrameOffset(), lastFrame + def.getFrameOffset(), maskHostBuffer,
                            readFrameTime, true, readFrameCalls, readFrameExits));
  }

  virtual Input::ProbeResult probe(const Ptv::Value & /*config*/) const {
    ENSURE(false, "not supported");
    return Input::ProbeResult({false, false, -1, -1, -1, -1});
  }

  virtual int getFirstFrame() const { return firstFrame; }

  virtual int getNumFrames() const { return lastFrame; }

 private:
  FrameRate frameRate;
  frameid_t firstFrame;
  frameid_t lastFrame;
  const unsigned char *maskHostBuffer;
  std::atomic<int> *readFrameCalls;
  std::atomic<int> *readFrameExits;
  int readFrameTime;
};

/**
 * A fake reader that notifies on deallocation
 */
class ResourceCheckReader : public Input::VideoReader {
 public:
  explicit ResourceCheckReader(std::atomic<int> *dealloc)
      : Reader(0), VideoReader(1, 1, 1, PixelFormat::RGBA, Host, {1, 1}, 0, 1, false, nullptr), dealloc(dealloc) {}

  virtual ~ResourceCheckReader() { (*dealloc)++; }

  virtual Input::ReadStatus readFrame(mtime_t &, unsigned char *) {
    return Status{Origin::Input, ErrType::UnsupportedAction, "ResourceCheckReader doesn't implement readFrame"};
  }

  virtual Status seekFrame(frameid_t) {
    return {Origin::Input, ErrType::UnsupportedAction, "ResourceCheckReader doesn't implement seekFrame"};
  }

 private:
  std::atomic<int> *dealloc;
};

/**
 * A reader factory that provides the ability to check
 * that all of its created readers are destroyed once
 */
class ResourceCheckReaderFactory : public Input::ReaderFactory {
 public:
  ResourceCheckReaderFactory(std::vector<std::atomic<int> *> &readerChecks, std::atomic<int> &factoryCheck)
      : readerChecks(readerChecks), factoryCheck(factoryCheck) {}

  virtual ~ResourceCheckReaderFactory() { factoryCheck++; }

  virtual Potential<Input::Reader> create(readerid_t, const Core::ReaderInputDefinition &) const {
    auto counter = new std::atomic<int>(0);
    readerChecks.push_back(counter);
    return Potential<Input::Reader>(new ResourceCheckReader(counter));
  }

  virtual Input::ProbeResult probe(const Ptv::Value & /*config*/) const {
    ENSURE(false, "not supported");
    return Input::ProbeResult({false, false, -1, -1, -1, -1});
  }

  virtual int getFirstFrame() const { return 0; }

  virtual int getNumFrames() const { return -1; }

 private:
  std::vector<std::atomic<int> *> &readerChecks;
  std::atomic<int> &factoryCheck;
};

}  // namespace Testing
}  // namespace VideoStitch
