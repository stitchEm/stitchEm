// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "orahProcessor.hpp"
#include "audio/converter.hpp"
#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Audio {
static const std::string o2bTag = "o2b";

OrahProcessor::OrahProcessor() : AudioObject("orah2b", AudioFunction::PROCESSOR) {
  orah4i2b_ = Orah4i::Orah4iToB::create();
}

OrahProcessor::~OrahProcessor() {}

void OrahProcessor::step(AudioBlock &out, const AudioBlock &in) {
  audioSample_t *inInterleaved = new audioSample_t[in.numSamples() * getNbChannelsFromChannelLayout(in.getLayout())];
  audioSample_t *outInterleaved = new audioSample_t[in.numSamples() * getNbChannelsFromChannelLayout(in.getLayout())];
  convertAudioBlockToInterleavedSamples(in, inInterleaved);
  orah4i2b_->process(inInterleaved, outInterleaved);
  convertInterleavedSamplesToAudioBlock(outInterleaved, (int)in.numSamples(), AMBISONICS_WXYZ, out);
  out.setTimestamp(in.getTimestamp());
  delete[] inInterleaved;
  delete[] outInterleaved;
}

void OrahProcessor::step(AudioBlock &inout) { step(inout, inout); }

}  // namespace Audio
}  // namespace VideoStitch
