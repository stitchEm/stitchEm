// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <vector>
#include <cmath>

#include "libvideostitch/audioObject.hpp"
#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Audio {

typedef double gainDB_t;
typedef double gainLin_t;
typedef double gainPercent_t;

static const gainDB_t kGainMax = 20.;
static const gainDB_t kGainMin = -100.;

gainLin_t dBToLin(gainDB_t x);
gainPercent_t dBToPercent(gainDB_t x);

gainDB_t linToDB(gainLin_t x);
gainPercent_t linToPercent(gainLin_t x);

gainDB_t percentToDB(gainPercent_t x);
gainLin_t percentToLin(gainPercent_t x);

class Gain : public AudioObject {
 public:
  explicit Gain(gainDB_t gaindB, bool reversePolarity, bool mute);

  gainDB_t getGainDB();
  gainPercent_t getGainPercent();
  bool getMute();
  bool getReversePolarity();

  void setGainDB(gainDB_t gain);
  void setGainPercent(gainPercent_t gain);
  void setMute(bool m);
  void setReversePolarity(bool r);
  void step(AudioBlock &, const AudioBlock &) override;
  void step(AudioBlock &inout) override;

 private:
  gainLin_t gain_;
  bool reversePolarity_;
  bool mute_;
};

}  // namespace Audio
}  // namespace VideoStitch
