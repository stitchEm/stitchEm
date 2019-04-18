// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Ambisonic decoder coefficients parser

#include "libvideostitch/ambDecoderDef.hpp"
#include "libvideostitch/ambisonic.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "parse/json.hpp"

#include <sstream>

namespace VideoStitch {
namespace Audio {

AmbisonicDecoderDef::AmbisonicDecoderDef(const Ptv::Value& value)
    : _order(getStringFromAmbisonicOrder(AmbisonicOrder::FIRST_ORDER)), _isValid(false) {
  setCoef(value);
}

void AmbisonicDecoderDef::setCoef(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("AmbisonicDecoderDef", value, Ptv::Value::OBJECT)) {
    _isValid = false;
    return;
  }

  if (Parse::populateString("AmbisonicDecoderDef", value, "order", _order, true) == Parse::PopulateResult_Ok &&
      getAmbisonicOrderFromString(_order) != AmbisonicOrder::UNKNOWN) {
    Logger::get(Logger::Warning) << "[ambisonic-decoder] Order " << _order << " not managed. Try a first order"
                                 << std::endl;
    _order = getStringFromAmbisonicOrder(AmbisonicOrder::FIRST_ORDER);
  }

  if (value.has("coefficients") && value.has("coefficients")->has(getStringFromChannelLayout(STEREO))) {
    const Ptv::Value* stereoCoef = value.has("coefficients")->has(getStringFromChannelLayout(STEREO));
    parseFirstOrderCoef(stereoCoef, STEREO, SPEAKER_FRONT_LEFT);
    parseFirstOrderCoef(stereoCoef, STEREO, SPEAKER_FRONT_RIGHT);
    _isValid = true;
  }

  if (value.has("coefficients") && value.has("coefficients")->has(getStringFromChannelLayout(_5POINT1))) {
    const Ptv::Value* fivePoint1 = value.has("coefficients")->has(getStringFromChannelLayout(_5POINT1));
    parseFirstOrderCoef(fivePoint1, _5POINT1, SPEAKER_FRONT_LEFT);
    parseFirstOrderCoef(fivePoint1, _5POINT1, SPEAKER_FRONT_RIGHT);
    parseFirstOrderCoef(fivePoint1, _5POINT1, SPEAKER_LOW_FREQUENCY);
    parseFirstOrderCoef(fivePoint1, _5POINT1, SPEAKER_FRONT_CENTER);
    parseFirstOrderCoef(fivePoint1, _5POINT1, SPEAKER_SIDE_LEFT);
    parseFirstOrderCoef(fivePoint1, _5POINT1, SPEAKER_SIDE_RIGHT);
    _isValid = true;
  }

  // TODO: add other supported layouts 7.1 and quad stereo
}

AmbisonicDecoderDef::~AmbisonicDecoderDef() {}

const ambCoefTable_t& AmbisonicDecoderDef::getCoefficients() const { return _coefs; }

PotentialValue<channelCoefTable_t> AmbisonicDecoderDef::getCoefficientsByLayout(ChannelLayout layout) const {
  if (_coefs.find(layout) == _coefs.end()) {
    std::stringstream ss;
    ss << "[ambisonic-decoder-coef] Layout " << getStringFromChannelLayout(layout) << "Not managed";
    return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, ss.str());
  }
  return _coefs.at(layout);
}

void AmbisonicDecoderDef::parseFirstOrderCoef(const Ptv::Value* v, ChannelLayout layout, ChannelMap out) {
  if (v->has(getStringFromChannelType(out))) {
    const Ptv::Value* coef = v->has(getStringFromChannelType(out));
    for (int64_t c = SPEAKER_AMB_W; c <= SPEAKER_AMB_Z; c = c << 1) {
      if (coef->has(getStringFromChannelType((ChannelMap)c))) {
        _coefs[layout][out][(ChannelMap)c] = coef->has(getStringFromChannelType((ChannelMap)c))->asDouble();
      }
    }
  }
}

Ptv::Value* AmbisonicDecoderDef::serialize() const {
  std::unique_ptr<Ptv::Value> res(Ptv::Value::emptyObject());
  res->push("order", new Parse::JsonValue(_order));
  // Serialize coefficients
  Ptv::Value* tableCoefs = Ptv::Value::emptyObject();
  for (auto& tablePerLayout : _coefs) {
    Ptv::Value* tableLayout = Ptv::Value::emptyObject();
    for (auto& tablePerOutChannel : tablePerLayout.second) {
      Ptv::Value* tableOutChannel = Ptv::Value::emptyObject();
      for (auto& tablePerAmbChannel : tablePerOutChannel.second) {
        tableOutChannel->push(getStringFromChannelType(tablePerAmbChannel.first),
                              new Parse::JsonValue(tablePerAmbChannel.second));
      }
      tableLayout->push(getStringFromChannelType(tablePerOutChannel.first), tableOutChannel);
    }
    tableCoefs->push(getStringFromChannelLayout(tablePerLayout.first), tableLayout);
  }
  res->push("coefficients", tableCoefs);
  return res.release();
}

AmbisonicDecoderDef* AmbisonicDecoderDef::clone() {
  AmbisonicDecoderDef* result = new AmbisonicDecoderDef(_coefs, _order);
  return result;
}

AmbisonicDecoderDef::AmbisonicDecoderDef(const ambCoefTable_t& coefs, const std::string& order)
    : _coefs(coefs), _order(order), _isValid(true) {}

}  // namespace Audio
}  // namespace VideoStitch
