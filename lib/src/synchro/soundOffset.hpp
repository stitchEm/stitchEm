// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audioBlock.hpp"
#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/frame.hpp"

#include <memory>

namespace VideoStitch {

namespace Input {
class Reader;
}

namespace Synchro {

struct audioSyncResult_t {
  int nSources;
  std::vector<std::vector<double>> timeOffset;
  std::vector<std::vector<double>> corrVal;
};

/**
 * An algorithm that aligns inputs using their sound.
 */
class SoundOffsetAlignAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;

  explicit SoundOffsetAlignAlgorithm(const Ptv::Value* config);
  virtual ~SoundOffsetAlignAlgorithm() {}

  Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                              Util::OpaquePtr** ctx = NULL) const override;

 protected:
  uint64_t firstFrame;
  uint64_t lastFrame;

  Status doAlign(const Core::PanoDefinition& pano, std::vector<int>& frames, ProgressReporter* progress) const;

  /* Find offsets - Ooura SG_H FFT */
  static Status foffs_sgh(std::vector<Audio::AudioBlock>& in, int fs, audioSyncResult_t& res,
                          ProgressReporter* progress);

 private:
  // Initialize the audio & video readers, find frame and samplingn rate
  Status setupReaders(const Core::PanoDefinition& pano, ProgressReporter* progress, FrameRate& frameRate,
                      Audio::SamplingRate& sampleRate, std::vector<std::unique_ptr<Input::Reader>>& readers) const;

  // Read all audio samples from firstFrame to lastFrame into samplesToRead
  Status readSamples(const std::vector<std::unique_ptr<Input::Reader>>& readers, const FrameRate& frameRate,
                     ProgressReporter* progress, std::vector<mtime_t>& initialPos,
                     std::vector<Audio::AudioBlock>& samplesToRead) const;
};

}  // namespace Synchro
}  // namespace VideoStitch
