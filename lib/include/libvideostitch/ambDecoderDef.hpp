// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Ambisonic decoder coefficients parser

#pragma once

#include "object.hpp"
#include "status.hpp"
#include "audio.hpp"

namespace VideoStitch {

namespace Ptv {
class Value;
}

namespace Audio {

typedef std::map<ChannelMap, std::map<ChannelMap, double>> channelCoefTable_t;
typedef std::map<ChannelLayout, channelCoefTable_t> ambCoefTable_t;

class AmbisonicDecoderDef : public Ptv::Object {
 public:
  explicit AmbisonicDecoderDef(const Ptv::Value& value);
  ~AmbisonicDecoderDef();

  const ambCoefTable_t& getCoefficients() const;
  PotentialValue<channelCoefTable_t> getCoefficientsByLayout(ChannelLayout layout) const;
  Ptv::Value* serialize() const;
  AmbisonicDecoderDef* clone();

  void setCoef(const Ptv::Value& value);

 private:
  AmbisonicDecoderDef(const ambCoefTable_t& coefs, const std::string& order);
  void parseFirstOrderCoef(const Ptv::Value* v, ChannelLayout layout, ChannelMap out);
  ambCoefTable_t _coefs;
  std::string _order;
  bool _isValid;
};

}  // namespace Audio
}  // namespace VideoStitch
