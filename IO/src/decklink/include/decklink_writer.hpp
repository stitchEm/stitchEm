// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/stitchOutput.hpp"

#if defined(_WIN32)
#include "DeckLinkAPI_h.h"
#else
#include "DeckLinkAPI.h"
#include "DeckLinkAPIModes.h"
#endif

/**
 * DeckLink Mini Monitor
 */
namespace VideoStitch {
namespace Output {

class DeckLinkWriter : public VideoWriter, public AudioWriter {
 public:
  virtual ~DeckLinkWriter();

  static DeckLinkWriter* create(const Ptv::Value* config, const std::string& name, unsigned width, unsigned height,
                                FrameRate framerate, const Audio::SamplingDepth depth,
                                const Audio::ChannelLayout layout);
  static bool handles(const Ptv::Value* config);

  virtual void pushVideo(const Frame& videoFrame);
  virtual void pushAudio(Audio::Samples& audioSamples);

 private:
  DeckLinkWriter(const std::string& name, unsigned width, unsigned height, FrameRate fps, size_t frameSize,
                 const Audio::SamplingDepth depth, const Audio::ChannelLayout layout,
                 std::shared_ptr<IDeckLink> subDevice, std::shared_ptr<IDeckLinkConfiguration> configuration,
                 std::shared_ptr<IDeckLinkConfiguration> configurationForHalfDuplex, std::shared_ptr<IDeckLinkOutput>,
                 std::shared_ptr<IDeckLinkMutableVideoFrame> outputFrame);

  std::shared_ptr<IDeckLink> subDevice;
  std::shared_ptr<IDeckLinkConfiguration> configuration;  // We need to keep the configuration because
  // In Decklink SDK doc: "Changes will persist until the IDeckLinkConfiguration object is released"
  std::shared_ptr<IDeckLinkConfiguration>
      configurationForHalfDuplex;  // Same need here but the configuration is not necessarily
  // the one of the above sub device. It can be the configuration of the paired sub device (for Quad 2 and Duo 2)
  std::shared_ptr<IDeckLinkOutput> output;
  std::shared_ptr<IDeckLinkMutableVideoFrame> outputFrame;
  size_t frameSize;

  bool firstFrame = true;

  static const BMDVideoOutputFlags outputFlags;
  static const BMDFrameFlags outputFrameFlags;
};

}  // namespace Output
}  // namespace VideoStitch
