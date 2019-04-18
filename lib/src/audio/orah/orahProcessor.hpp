// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "orah4i2b/header/orah4i2b.hpp"
#include "libvideostitch/audioObject.hpp"

namespace VideoStitch {
namespace Audio {

class OrahProcessor : public AudioObject {
 public:
  explicit OrahProcessor();
  ~OrahProcessor();

  void step(AudioBlock &out, const AudioBlock &in);
  void step(AudioBlock &inout);

 private:
  Orah4i::Orah4iToB *orah4i2b_;
};

}  // namespace Audio
}  // namespace VideoStitch
