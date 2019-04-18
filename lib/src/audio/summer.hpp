// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
//  A class to sum audio blocks

#pragma once

#include "libvideostitch/audioBlock.hpp"

#include <sstream>

namespace VideoStitch {
namespace Audio {

Status sum(std::vector<AudioBlock> &blocks, AudioBlock &output) {
  ChannelLayout layout = blocks.begin()->getLayout();
  size_t nbSamples = blocks.begin()->numSamples();
  for (const AudioBlock &block : blocks) {
    if (block.getLayout() != layout) {
      std::stringstream errMsg;
      errMsg << "Cannot sum blocks with different layouts: expected \"" << getStringFromChannelLayout(layout)
             << "\" got \"" << getStringFromChannelLayout(block.getLayout()) << "\"";
      return {Origin::AudioPipeline, ErrType::AlgorithmFailure, errMsg.str()};
    }
    if (block.numSamples() != nbSamples) {
      std::stringstream errMsg;
      errMsg << "Cannot sum blocks with different size: expected " << nbSamples << " got " << block.numSamples();
      return {Origin::AudioPipeline, ErrType::AlgorithmFailure, errMsg.str()};
    }
  }
  output.setChannelLayout(layout);
  output.resize(nbSamples);
  output.assign(nbSamples, 0.);

  for (AudioBlock &block : blocks) {
    output += block;
  }
  return Status::OK();
}

}  // namespace Audio
}  // namespace VideoStitch
