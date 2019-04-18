// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ajastuff/common/circularbuffer.h"
#include "ajastuff/system/thread.h"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"

#include "ntv2plugin.hpp"
#include <atomic>

namespace VideoStitch {
namespace Input {

class NTV2Reader : public VideoReader, public AudioReader {
 public:
  static NTV2Reader* create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height);

  virtual ~NTV2Reader();

  ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) override;
  ReadStatus readFrame(mtime_t&, unsigned char* video) override;
  Status seekFrame(frameid_t) override;
  Status seekFrame(mtime_t) override;
  size_t available() override;
  bool eos() override;

 private:
  NTV2Reader(readerid_t id, const int64_t width, const int64_t height, const UWord deviceIndex, const bool withAudio,
             const NTV2Channel channel, FrameRate fps, bool interlaced);

  // -- init
  AJAStatus init();
  void quit();
  AJAStatus setupVideo(NTV2Channel);
  AJAStatus setupAudio();
  void setupHostBuffers();
  void routeInputSignal(NTV2Channel);
  AJAStatus run();

  // -- capture
  AJAThread* producerThread;
  void startProducerThread();
  void captureFrames();
  static void producerThreadStatic(AJAThread*, void*);
  bool InputSignalHasTimecode() const;

  const uint32_t deviceIndex;

  const bool withAudio;
  const NTV2Channel inputChannel;
  NTV2InputSource inputSource;
  NTV2VideoFormat videoFormat;
  NTV2FrameRate frameRate;
  NTV2AudioSystem audioSystem;
  CNTV2SignalRouter router;
  int32_t startFrameId;
  NTV2TCIndex timeCodeSource;

  NTV2Device* device = nullptr;

  DisplayMode displayMode;
  FrameRate* frameRateVS = nullptr;

  std::atomic<bool> noSignal = true;
  bool interlaced;
  bool AJAStop;

  AUTOCIRCULATE_TRANSFER mInputTransfer; /*class use for autocircular, don't memset it to zero !*/

  std::atomic<bool> globalQuit;  /// Set "true" to gracefully stop

  uint32_t videoBufferSize;  /// in bytes
  uint32_t audioBufferSize;  /// in bytes
  mtime_t videoTS;

  AVDataBuffer aVHostBuffer[CIRCULAR_BUFFER_SIZE];
  AJACircularBuffer<AVDataBuffer*> aVCircularBuffer;

  typedef uint32_t ajasample_t;
  std::vector<ajasample_t> audioBuff;
  mtime_t audioTS;
  uint64_t nbSamplesRead;
  std::mutex audioBuffMutex;
  std::mutex quitMutex;
};

}  // namespace Input
}  // namespace VideoStitch
