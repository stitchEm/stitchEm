// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Ambisonic encoder and decoder

#include "libvideostitch/ambisonic.hpp"
#include "libvideostitch/logging.hpp"

#include <cmath>

static std::string ambTag("ambisonic");

namespace VideoStitch {
namespace Audio {

int getNbAmbisonicChannelsFromOrder(AmbisonicOrder order) {
  switch (order) {
    case AmbisonicOrder::FIRST_ORDER:
      return 4;
    case AmbisonicOrder::SECOND_ORDER:
      return 9;
    case AmbisonicOrder::THIRD_ORDER:
      return 16;
    case AmbisonicOrder::UNKNOWN:
      return 0;
  }
  return 0;
}

const char* getStringFromAmbisonicNorm(AmbisonicNorm norm) {
  switch (norm) {
    case AmbisonicNorm::FUMA:
      return "FUMA";
    case AmbisonicNorm::SN3D:
      return "SN3D";
  }
  return "";
}

const char* getStringFromAmbisonicOrder(AmbisonicOrder order) {
  switch (order) {
    case AmbisonicOrder::FIRST_ORDER:
      return "first_order";
    case AmbisonicOrder::SECOND_ORDER:
      return "second_order";
    case AmbisonicOrder::THIRD_ORDER:
      return "third_order";
    case AmbisonicOrder::UNKNOWN:
      return "unknown_order";
  }
  return "";
}

ChannelLayout getChannelLayoutFromAmbisonicOrder(AmbisonicOrder order) {
  switch (order) {
    case AmbisonicOrder::FIRST_ORDER:
      return AMBISONICS_WXYZ;
    case AmbisonicOrder::SECOND_ORDER:
      return AMBISONICS_2ND;
    case AmbisonicOrder::THIRD_ORDER:
      return AMBISONICS_3RD;
    case AmbisonicOrder::UNKNOWN:
      return UNKNOWN;
  }
  return UNKNOWN;
}

ChannelMap getChannelAmbFromChanIndex(int i) {
  switch (i) {
    case 0:
      return SPEAKER_AMB_W;
    case 1:
      return SPEAKER_AMB_Y;
    case 2:
      return SPEAKER_AMB_Z;
    case 3:
      return SPEAKER_AMB_X;
    case 4:
      return SPEAKER_AMB_V;
    case 5:
      return SPEAKER_AMB_T;
    case 6:
      return SPEAKER_AMB_R;
    case 7:
      return SPEAKER_AMB_S;
    case 8:
      return SPEAKER_AMB_U;
    case 9:
      return SPEAKER_AMB_Q;
    case 10:
      return SPEAKER_AMB_O;
    case 11:
      return SPEAKER_AMB_M;
    case 12:
      return SPEAKER_AMB_K;
    case 13:
      return SPEAKER_AMB_L;
    case 14:
      return SPEAKER_AMB_N;
    case 15:
      return SPEAKER_AMB_P;
    default:
      return NO_SPEAKER;
  }
}

AmbEncoder::AmbEncoder(AmbisonicOrder order, AmbisonicNorm norm)
    : AudioObject("AmbEncoder", AudioFunction::PROCESSOR), _order(order), _norm(norm) {
  _coefficients[MONO][SPEAKER_FRONT_LEFT] = makeMonoCoef({0, 0}, _order, _norm);
  initializeStereoCoef();
  initialize51Coef();
  initialize71Coef();
}

AmbEncoder::~AmbEncoder() {}

void AmbEncoder::setMonoSourcePosition(const AngularPosition& pos) { updateCoef(pos, MONO, SPEAKER_FRONT_LEFT); }

void AmbEncoder::step(AudioBlock& out, const AudioBlock& in) {
  out.setChannelLayout(getChannelLayoutFromAmbisonicOrder(_order));
  out.setTimestamp(in.getTimestamp());
  out.assign(in.numSamples(), 0.);

  int nbAmbChannels = getNbChannelsFromChannelLayout(getChannelLayoutFromAmbisonicOrder(_order));
  for (int iAmb = 0; iAmb < nbAmbChannels; iAmb++) {
    auto chanMap = getChannelAmbFromChanIndex(iAmb);
    auto& ambiTrack = out[chanMap];
    for (const AudioTrack& track : in) {
      auto coef = _coefficients[in.getLayout()][track.channel()][chanMap];
      for (size_t i = 0; i < in.numSamples(); ++i) {
        ambiTrack[i] += track[i] * coef;
      }
    }
  }
}

void AmbEncoder::step(AudioBlock& inout) {
  size_t numSamples = inout.numSamples();
  int nbAmbChannels = getNbChannelsFromChannelLayout(getChannelLayoutFromAmbisonicOrder(_order));

  for (int iAmb = 0; iAmb < nbAmbChannels; iAmb++) {
    auto chanMap = getChannelAmbFromChanIndex(iAmb);
    auto& ambiTrack = inout[chanMap];
    ambiTrack.assign(numSamples, 0.);
    for (const AudioTrack& track : inout) {
      auto coef = _coefficients[inout.getLayout()][track.channel()][chanMap];
      for (size_t i = 0; i < numSamples; ++i) {
        ambiTrack[i] += track[i] * coef;
      }
    }
  }
  inout.setChannelLayout(getChannelLayoutFromAmbisonicOrder(_order));
}

#define COS2(x) (pow(cos(x), 2.))
#define SIN2(x) (pow(sin(x), 2.))
#define COS3(x) (pow(cos(x), 3.))

// Equations from the "Space in Music - Music in Space"  an Mphil thesis by Dave Malham,
// submitted to the University of York in April 2003
#define COMPUTE_FIRST_ORDER_COEF(c, az, el) \
  {                                         \
    c[SPEAKER_AMB_W] = 1.;                  \
    c[SPEAKER_AMB_Y] = sin(az) * cos(el);   \
    c[SPEAKER_AMB_Z] = sin(el);             \
    c[SPEAKER_AMB_X] = cos(az) * cos(el);   \
  }

#define COMPUTE_2ND_ORDER_COEF(c, az, el)                       \
  {                                                             \
    COMPUTE_FIRST_ORDER_COEF(c, az, el);                        \
    c[SPEAKER_AMB_V] = sqrt(3.) / 2. * sin(2. * az) * COS2(el); \
    c[SPEAKER_AMB_T] = sqrt(3.) / 2. * sin(az) * sin(2 * el);   \
    c[SPEAKER_AMB_R] = (3. * SIN2(el) - 1.) / 2.;               \
    c[SPEAKER_AMB_S] = sqrt(3.) / 2. * cos(az) * sin(2 * el);   \
    c[SPEAKER_AMB_U] = sqrt(3.) / 2. * cos(2. * az) * COS2(el); \
  }

#define COMPUTE_3RD_ORDER_COEF(c, az, el)                                       \
  {                                                                             \
    COMPUTE_2ND_ORDER_COEF(c, az, el);                                          \
    c[SPEAKER_AMB_Q] = sqrt(5.) / 8. * sin(3 * az) * COS3(el);                  \
    c[SPEAKER_AMB_O] = sqrt(15.) / 2. * sin(2. * az) * sin(el) * COS2(el);      \
    c[SPEAKER_AMB_M] = sqrt(3.) / 8. * sin(az) * cos(el) * (5 * SIN2(el) - 1.); \
    c[SPEAKER_AMB_K] = sin(el) * (5 * pow(sin(el), 2.) - 3.) / 2.;              \
    c[SPEAKER_AMB_L] = sqrt(3.) / 8. * cos(az) * cos(el) * (5 * SIN2(el) - 1.); \
    c[SPEAKER_AMB_N] = sqrt(15.) / 2. * cos(2. * az) * sin(el) * COS2(el);      \
    c[SPEAKER_AMB_P] = sqrt(5.) / 8. * cos(3 * az) * COS3(el);                  \
  }

// Conversion table for FUMA weights
const std::map<ChannelMap, double> fumaWeigths = {
    // 1st order coefficients
    {SPEAKER_AMB_W, 1. / sqrt(2.)},
    {SPEAKER_AMB_X, 1.},
    {SPEAKER_AMB_Y, 1.},
    {SPEAKER_AMB_Z, 1.},
    // 2nd order coefficients
    {SPEAKER_AMB_R, 1.},
    {SPEAKER_AMB_S, 2. / sqrt(3.)},
    {SPEAKER_AMB_T, 2 / sqrt(3.)},
    {SPEAKER_AMB_U, 2 / sqrt(3.)},
    {SPEAKER_AMB_V, 2 / sqrt(3.)},
    // 3rd order coefficients
    {SPEAKER_AMB_K, 1.},
    {SPEAKER_AMB_L, sqrt(45.) / 32.},
    {SPEAKER_AMB_M, sqrt(45.) / 32.},
    {SPEAKER_AMB_N, 3. / sqrt(5.)},
    {SPEAKER_AMB_O, 3. / sqrt(5.)},
    {SPEAKER_AMB_P, sqrt(8.) / 5.},
    {SPEAKER_AMB_Q, sqrt(8.) / 5.}};

void convertToFuma(std::map<ChannelMap, double>& coef) {
  for (auto& kvPerAmbCh : coef) {
    kvPerAmbCh.second *= fumaWeigths.at(kvPerAmbCh.first);
  }
}

std::map<ChannelMap, double> AmbEncoder::makeMonoCoef(const AngularPosition& coord, AmbisonicOrder order,
                                                      AmbisonicNorm norm) {
  std::map<ChannelMap, double> coef;
  switch (order) {
    case AmbisonicOrder::FIRST_ORDER:
      COMPUTE_FIRST_ORDER_COEF(coef, coord.az, coord.el);
      break;
    case AmbisonicOrder::SECOND_ORDER:
      COMPUTE_2ND_ORDER_COEF(coef, coord.az, coord.el);
      break;
    case AmbisonicOrder::THIRD_ORDER:
      COMPUTE_3RD_ORDER_COEF(coef, coord.az, coord.el);
      break;
    case AmbisonicOrder::UNKNOWN:
      Logger::get(Logger::Verbose) << "Ambisonic " << getStringFromAmbisonicOrder(order) << " not managed yet."
                                   << std::endl;
  }

  if (norm == AmbisonicNorm::FUMA) {
    convertToFuma(coef);
  }

  return coef;
}

void AmbEncoder::showCoef() const {
  for (const auto& kvPerLayout : _coefficients) {
    for (const auto& kvPerChannel : kvPerLayout.second) {
      std::string channel = getStringFromChannelType(kvPerChannel.first);
      for (const auto& kvPerAmbChannel : kvPerChannel.second) {
        std::string ambChannel = getStringFromChannelType(kvPerAmbChannel.first);
        Logger::verbose(ambTag) << "coef for channel " << channel << " to " << ambChannel << " "
                                << kvPerAmbChannel.second << std::endl;
      }
    }
  }
}

std::map<ChannelMap, double> AmbEncoder::makeSubWooferCoef(AmbisonicOrder order, AmbisonicNorm norm) {
  std::map<ChannelMap, double> coef;
  coef[SPEAKER_AMB_W] = 1.;
  ChannelMap ambCh = SPEAKER_AMB_W;
  for (int i = 0; i < getNbAmbisonicChannelsFromOrder(order); ++i) {
    ambCh = static_cast<ChannelMap>(static_cast<int64_t>(ambCh) << 1);
    if (ambCh == SPEAKER_AMB_W) {
      coef[ambCh] = 1.;
    } else {
      coef[ambCh] = 0.;
    }
  }

  if (norm == AmbisonicNorm::FUMA) {
    convertToFuma(coef);
  }
  return coef;
}

void AmbEncoder::updateCoef(AngularPosition pos, ChannelLayout layout, ChannelMap m) {
  switch (_order) {
    case AmbisonicOrder::FIRST_ORDER:
      COMPUTE_FIRST_ORDER_COEF(_coefficients.at(layout).at(m), pos.az, pos.el);
      break;
    case AmbisonicOrder::SECOND_ORDER:
      COMPUTE_2ND_ORDER_COEF(_coefficients.at(layout).at(m), pos.az, pos.el);
      break;
    case AmbisonicOrder::THIRD_ORDER:
      COMPUTE_3RD_ORDER_COEF(_coefficients.at(layout).at(m), pos.az, pos.el);
      break;
    case AmbisonicOrder::UNKNOWN:
      Logger::error(ambTag) << " order " << getStringFromAmbisonicOrder(_order) << " not managed" << std::endl;
  }
}

void AmbEncoder::initializeStereoCoef() {
  AngularPosition leftCoord = {M_PI / 2., 0.};
  AngularPosition rightCoord = {-M_PI / 2., 0.};
  _coefficients[STEREO][SPEAKER_FRONT_LEFT] = makeMonoCoef(leftCoord, _order, _norm);
  _coefficients[STEREO][SPEAKER_FRONT_RIGHT] = makeMonoCoef(rightCoord, _order, _norm);
}

void AmbEncoder::initialize51Coef() {
  AngularPosition centerCoord = {0., 0.};
  AngularPosition frontLeftCoord = {30. * M_PI / 180., 0.};
  AngularPosition frontRightCoord = {330. * M_PI / 180., 0.};
  AngularPosition sideLeftCoord = {110. * M_PI / 180., 0.};
  AngularPosition sideRightCoord = {250. * M_PI / 180., 0.};
  _coefficients[_5POINT1][SPEAKER_FRONT_LEFT] = makeMonoCoef(frontLeftCoord, _order, _norm);
  _coefficients[_5POINT1][SPEAKER_FRONT_RIGHT] = makeMonoCoef(frontRightCoord, _order, _norm);
  _coefficients[_5POINT1][SPEAKER_FRONT_CENTER] = makeMonoCoef(centerCoord, _order, _norm);
  _coefficients[_5POINT1][SPEAKER_LOW_FREQUENCY] = makeSubWooferCoef(_order, _norm);
  _coefficients[_5POINT1][SPEAKER_SIDE_LEFT] = makeMonoCoef(sideLeftCoord, _order, _norm);
  _coefficients[_5POINT1][SPEAKER_SIDE_RIGHT] = makeMonoCoef(sideRightCoord, _order, _norm);
}

void AmbEncoder::initialize71Coef() {
  initializeStereoCoef();
  AngularPosition centerCoord = {0., 0.};
  AngularPosition sideLeftCoord = {M_PI / 2., 0.};
  AngularPosition sideRightCoord = {-M_PI / 2., 0.};
  AngularPosition backLeftCoord = {150. * M_PI / 180., 0.};
  AngularPosition backRightCoord = {-150. * M_PI / 180., 0.};
  _coefficients[_7POINT1][SPEAKER_FRONT_CENTER] = makeMonoCoef(centerCoord, _order, _norm);
  _coefficients[_7POINT1][SPEAKER_LOW_FREQUENCY] = makeSubWooferCoef(_order, _norm);
  _coefficients[_7POINT1][SPEAKER_SIDE_LEFT] = makeMonoCoef(sideLeftCoord, _order, _norm);
  _coefficients[_7POINT1][SPEAKER_SIDE_RIGHT] = makeMonoCoef(sideRightCoord, _order, _norm);
  _coefficients[_7POINT1][SPEAKER_BACK_LEFT] = makeMonoCoef(backLeftCoord, _order, _norm);
  _coefficients[_7POINT1][SPEAKER_BACK_RIGHT] = makeMonoCoef(backRightCoord, _order, _norm);
}

AmbDecoder::AmbDecoder(ChannelLayout l, const ambCoefTable_t& coef)
    : AudioObject("AmbDecoder", AudioFunction::PROCESSOR), _outLayout(l), _coefficients(coef) {
  if (_coefficients.find(_outLayout) == _coefficients.end() ||
      getNbChannelsFromChannelLayout(_outLayout) != (int)_coefficients.at(_outLayout).size()) {
    Logger::warning(ambTag) << "Decoder bad coefficients given for " << getStringFromChannelLayout(_outLayout)
                            << std::endl;
  }
}

AmbDecoder::~AmbDecoder() {}

void AmbDecoder::step(AudioBlock& out, const AudioBlock& in) {
  if (in.getLayout() != AMBISONICS_WXYZ || _coefficients.find(_outLayout) == _coefficients.end()) {
    return;
  }
  out.setChannelLayout(_outLayout);
  out.setTimestamp(in.getTimestamp());
  size_t numSamples = in.numSamples();
  out.assign(numSamples, 0.);
  for (auto& outTrack : out) {
    for (const AudioTrack& inTrack : in) {
      auto coef = _coefficients[out.getLayout()][outTrack.channel()][inTrack.channel()];
      for (size_t i = 0; i < numSamples; ++i) {
        outTrack[i] += inTrack[i] * coef;
      }
    }
  }
}

void AmbDecoder::step(AudioBlock& inout) {
  if (inout.getLayout() != AMBISONICS_WXYZ || _coefficients.find(_outLayout) == _coefficients.end()) {
    return;
  }
  size_t numSamples = inout.numSamples();
  ChannelMap outCh = SPEAKER_FRONT_LEFT;
  while (outCh < NO_SPEAKER) {
    if (outCh & _outLayout) {
      inout[outCh].assign(numSamples, 0.);
      for (const AudioTrack& inTrack : inout) {
        auto coef = _coefficients[_outLayout][outCh][inTrack.channel()];
        for (size_t i = 0; i < numSamples; ++i) {
          inout[outCh][i] += inTrack[i] * coef;
        }
      }
    }
    outCh = static_cast<ChannelMap>(static_cast<int64_t>(outCh) << 1);
  }
  inout.setChannelLayout(_outLayout);
}

void AmbDecoder::setCoefficients(const ambCoefTable_t& coefficients) { _coefficients = coefficients; }

AmbisonicOrder getAmbisonicOrderFromString(const std::string& s) {
  if (getStringFromAmbisonicOrder(AmbisonicOrder::FIRST_ORDER) == s) {
    return AmbisonicOrder::FIRST_ORDER;
  } else if (getStringFromAmbisonicOrder(AmbisonicOrder::SECOND_ORDER) == s) {
    return AmbisonicOrder::SECOND_ORDER;
  } else if (getStringFromAmbisonicOrder(AmbisonicOrder::THIRD_ORDER) == s) {
    return AmbisonicOrder::THIRD_ORDER;
  } else {
    return AmbisonicOrder::UNKNOWN;
  }
}

AmbisonicOrder getAmbisonicOrderFromInt(int order) {
  switch (order) {
    case 1:
      return AmbisonicOrder::FIRST_ORDER;
    case 2:
      return AmbisonicOrder::SECOND_ORDER;
    case 3:
      return AmbisonicOrder::THIRD_ORDER;
    default:
      return AmbisonicOrder::UNKNOWN;
  }
}

/* Ambisonic rotator */
AmbRotator::AmbRotator(const AmbisonicOrder o)
    : AudioObject("ambRotator", AudioFunction::PROCESSOR), _order(o), _rotation({0, 0, 0}), _offset({0, 0, 0}) {
  Logger::verbose("ambrotator") << "Instatiate amb rotator" << std::endl;
}

AmbRotator::~AmbRotator() {}

void AmbRotator::setRotation(double yaw, double pitch, double roll) {
  Vector3<double> v(yaw, pitch, roll);
  std::lock_guard<std::mutex> lk(_rotMutex);
  _rotation = v;
}

void AmbRotator::setRotationOffset(double yaw, double pitch, double roll) {
  Vector3<double> v(yaw, pitch, roll);
  std::lock_guard<std::mutex> lk(_rotMutex);
  _offset = v;
}

void AmbRotator::step(AudioBlock& inOut) { step(inOut, inOut); }

static const channel_t kChanList[4] = {SPEAKER_AMB_W, SPEAKER_AMB_Y, SPEAKER_AMB_Z, SPEAKER_AMB_X};

#define ACN(buf__, acn__) buf__[kChanList[acn__]]

void AmbRotator::step(AudioBlock& out, const AudioBlock& in) {
  if (_order != AmbisonicOrder::FIRST_ORDER) {
    Logger::error("Only first-order rotations are implemented currently");
  }
  out.setChannelLayout(in.getLayout());
  out.resize(in.numSamples());
  std::unique_lock<std::mutex> lk(_rotMutex);
  double sinYaw = std::sin(_rotation(YAW) + _offset(YAW));
  double cosYaw = std::cos(_rotation(YAW) + _offset(YAW));
  double sinPitch = std::sin(_rotation(PITCH) + _offset(PITCH));
  double cosPitch = std::cos(_rotation(PITCH) + _offset(PITCH));
  double sinRoll = std::sin(_rotation(ROLL) + _offset(ROLL));
  double cosRoll = std::cos(_rotation(ROLL) + _offset(ROLL));
  lk.unlock();

  for (size_t i = 0; i < in.numSamples(); ++i) {
    audioSample_t save[2];
    // Yaw
    save[0] = ACN(in, 1)[i];
    save[1] = ACN(in, 3)[i];
    ACN(out, 0)[i] = ACN(in, 0)[i];
    ACN(out, 2)[i] = ACN(in, 2)[i];
    ACN(out, 1)[i] = (save[0] * cosYaw) + (save[1] * sinYaw);
    ACN(out, 3)[i] = (save[1] * cosYaw) - (save[0] * sinYaw);
    // Pitch
    save[0] = ACN(out, 2)[i];
    save[1] = ACN(out, 3)[i];
    ACN(out, 2)[i] = (save[0] * cosPitch) + (save[1] * sinPitch);
    ACN(out, 3)[i] = (save[1] * cosPitch) - (save[0] * sinPitch);
    // Roll
    save[0] = ACN(out, 1)[i];
    save[1] = ACN(out, 2)[i];
    ACN(out, 1)[i] = (save[0] * cosRoll) - (save[1] * sinRoll);
    ACN(out, 2)[i] = (save[0] * sinRoll) + (save[1] * cosRoll);
  }
}

}  // namespace Audio
}  // namespace VideoStitch
