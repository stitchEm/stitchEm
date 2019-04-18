// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ajastuff/common/circularbuffer.h"
#include "ajastuff/common/timecodeburn.h"

#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/circularBuffer.hpp"
#include "ntv2rp188.h"

#include "ntv2plugin.hpp"
#include <vector>
#include <mutex>
#include <atomic>

class AJAThread;

namespace VideoStitch {
namespace Output {

class NTV2Writer : public VideoWriter, public AudioWriter {
 public:
  static Output* create(const Ptv::Value& config, const std::string& name, const char* baseName, unsigned width,
                        unsigned height, FrameRate framerate);

  ~NTV2Writer();

  virtual void pushVideo(const Frame& videoFrame) override;
  virtual void pushAudio(Audio::Samples& audioSamples) override;

 private:
  NTV2Writer(const std::string& name, const UWord deviceIndex, const bool withAudio, const NTV2Channel channel,
             const NTV2VideoFormat format, unsigned width, unsigned height, unsigned offset_x, unsigned offset_y,
             FrameRate fps);

  // -- init
  AJAStatus _init();
  AJAStatus run();
  void quit();
  AJAStatus setupVideo(NTV2Channel);
  AJAStatus setupAudio();
  void setupHostBuffers();
  void setupOutputAutoCirculate();
  static bool checkChannelConf(unsigned width, unsigned height, int chan);
  void routeOutputSignal(NTV2Channel);

  AJA_PixelFormat getAJAPixelFormat(NTV2FrameBufferFormat format);
  bool outputDestHasRP188BypassEnabled(void);
  void disableRP188Bypass(void);
  /**
    @brief  Returns the RP188 DBB register number to use for the given NTV2OutputDestination.
    @param[in]  inOutputSource  Specifies the NTV2OutputDestination of interest.
    @return The number of the RP188 DBB register to use for the given output destination.
  **/
  static ULWord getRP188RegisterForOutput(const NTV2OutputDestination inOutputSource);

  // -- player
  void startConsumerThread();
  void startProducerThread();
  void playFrames();
  void produceFrames();
  static void consumerThreadStatic(AJAThread*, void*);
  static void producerThreadStatic(AJAThread*, void*);

  // Helper functions
  /**
   * @brief Initialize a table of tone per channel for 16 channels each channel has
   *        a specific frequency. Very useful to debug AJA output.
   *        Support 16 interleaved channels, int32_t sample format at 48 kHz
   *        Each tone is a multiple of 480 Hz
   **/
  void initSinTableFor16Channels();

  /**
   * @brief Fills the inout buffer with a tone per channel.
   *        Support 16 interleaved channels, int32_t sample format at 48 kHz
   *        Each tone is a multiple of 480 Hz
   **/
  uint32_t addAudioToneVS(int32_t* audioBuffer);

  AJAThread* consumerThread;
  AJAThread* producerThread;
  const uint32_t deviceIndex;
  uint8_t outputNb;
  const bool withAudio;
  const NTV2Channel outputChannel;
  NTV2OutputDestination outputDestination;
  NTV2VideoFormat videoFormat;
  NTV2AudioSystem audioSystem;  /// The audio system I'm using
  uint32_t nbAJAChannels;
  CNTV2SignalRouter router;

#ifdef DEPRECATED
  AUTOCIRCULATE_TRANSFER_STRUCT outputTransferStruct;  /// My A/C output transfer info
  AUTOCIRCULATE_TRANSFER_STATUS_STRUCT outputTransferStatusStruct;
#endif
  std::atomic<bool> globalQuit;  /// Set "true" to gracefully stop
  bool AJAStop;

  uint32_t videoBufferSize;  /// in bytes
  uint32_t audioBufferSize;  /// in bytes

  uint32_t nbSamplesPerFrame;

  AVDataBuffer aVHostBuffer[CIRCULAR_BUFFER_SIZE];
  AJACircularBuffer<AVDataBuffer*> aVCircularBuffer;

  CircularBuffer<uint8_t> videoBuffer;
  CircularBuffer<int32_t> audioBuffer;

  bool doLevelConversion;  /// Demonstrates a level A to level B conversion
  bool doMultiChannel;     /// Demonstrates how to configure the board for multi-format

  std::mutex frameMutex;

  unsigned offset_x;
  unsigned offset_y;

  int32_t preRollFrames;
  // Debug variables
  uint32_t producedFrames;
  uint32_t nbSamplesInWavForm;
  std::vector<int32_t> sinTable16Channels;
  ULWord currentSample;
};

}  // namespace Output
}  // namespace VideoStitch
