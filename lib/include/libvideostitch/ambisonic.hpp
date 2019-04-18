// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Ambisonic encoder and decoder

#pragma once

#include "audioObject.hpp"
#include "audioBlock.hpp"
#include "ambDecoderDef.hpp"

#include "matrix.hpp"

#include <mutex>

namespace VideoStitch {
namespace Audio {

// We manage only first order for the moment
enum class AmbisonicOrder {
  FIRST_ORDER,
  SECOND_ORDER,  // TODO
  THIRD_ORDER,   // TODO
  UNKNOWN
};

enum class AmbisonicNorm {
  SN3D,  // Prioritary
  FUMA   // In a second time
};

/**
 * @brief Angular coordinates
 */
struct AngularPosition {
  double az;  // in radians
  double el;  // in radians
};

/**
 * @brief Returns the number of ambisonic channels corresponding to the ambisonic order
 * @param order Ambisonic order
 */
VS_EXPORT int getNbAmbisonicChannelsFromOrder(AmbisonicOrder order);

/**
 * @brief Returns a string corresponding of the ambisonic normalization type
 * @param norm Type of the ambisonic normalization (FUMA or SN3D)
 */
VS_EXPORT const char *getStringFromAmbisonicNorm(AmbisonicNorm norm);

/**
 * @brief Returns a string corresponding of the ambisonic order
 * @param order Ambisonic order (FIRST_ORDER, SECOND_ORDER, THIRD_ORDER)
 */
VS_EXPORT const char *getStringFromAmbisonicOrder(AmbisonicOrder order);

/**
 * @brief Returns the ambisonic order corresponding to the string
 * @param string (FIRST_ORDER, SECOND_ORDER, THIRD_ORDER)
 */
VS_EXPORT AmbisonicOrder getAmbisonicOrderFromString(const std::string &s);

/**
 * @brief Returns the ambisonic order corresponding to `order`
 * @param order [1 .. 3]
 */
VS_EXPORT AmbisonicOrder getAmbisonicOrderFromInt(int order);

/**
 * @brief Returns a channel layout corresponding to the ambisonic order
 * @param order Ambisonic order (FIRST_ORDER, SECOND_ORDER, THIRD_ORDER)
 */
VS_EXPORT ChannelLayout getChannelLayoutFromAmbisonicOrder(AmbisonicOrder order);

/**
 * @brief Returns a channel ambisonic corresponding to the index
 *        Follow this standard: http://ambisonics.ch/standards/channels/
 * @param i Index (or
 */
VS_EXPORT ChannelMap getChannelAmbFromChanIndex(int i);

/**
 * Ambisonic encoder:
 * Implements encoder from the FIRST_ORDER to the THIRD_ORDER. For the moment only the FIRST_ORDER is officially
 * supported. The normalization type can be set to SN3D (default value) or FUMA. The encoder supports only the following
 * layouts:
 * - MONO: you can set a position in azimuth and elevation
 * - STEREO
 * - 5.1
 * - 7.1
 */
class VS_EXPORT AmbEncoder : public AudioObject {
 public:
  /**
   * @brief Constructor.
   * @param order ambisonic order
   * @param norm ambisonic normalization type FUMA or SN3D
   * @param inLayout layout of the source
   */
  AmbEncoder(AmbisonicOrder order, AmbisonicNorm norm);
  /**
   * @brief Default destructor.
   */
  ~AmbEncoder();

  /**
   * @brief Sets the position of the source and update the encoding coefficients accordingly
   * @param azimuth and elevation in radians
   */
  void setMonoSourcePosition(const AngularPosition &pos);
  void step(AudioBlock &out, const AudioBlock &in);
  void step(AudioBlock &inout);

 private:
  /**
   * @brief Initializes encoding coefficients for a stereo input
   */
  void initializeStereoCoef();

  /**
   * @brief Initializes encoding coefficients for a 5.1 input
   */
  void initialize51Coef();

  /**
   * @brief Initialize encoding coefficients for a 7.1 input
   */
  void initialize71Coef();

  /**
   * @brief Returns a vector of coefficients for a sub woofer channel
   *        The sub woofer channel is considered omnidirectional so it has speciacl encoding coefficients.
   *        for a first order (1,0,0,0)
   *        for a second order (1,0,0,0,0,0,0,0,0)
   *        for a third order (1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
   * @param order ambisonic order (FIRST_ORDER, SECOND_ORDER, THIRD_ORDER)
   * @param norm ambisonic normalization type (SN3D or FUMA)
   */
  std::map<ChannelMap, double> makeSubWooferCoef(AmbisonicOrder order, AmbisonicNorm norm);

  /**
   * @brief Initialize encoding coefficients for a mono source
   * @param coord azimuth and elevation in radians
   * @param map channel map of the concerned channel
   * @param order ambisonic order (FIRST_ORDER, SECOND_ORDER, THIRD_ORDER)
   * @param norm ambisonic normalization type (SN3D or FUMA)
   */
  std::map<ChannelMap, double> makeMonoCoef(const AngularPosition &coord, AmbisonicOrder order,
                                            AmbisonicNorm norm = AmbisonicNorm::SN3D);

  void showCoef() const;

  /**
   * @brief Updates the encoding coefficients with the new position
   * @param coord azimuth and elevation in radians
   * @param layout layout of the coefficients to update
   * @param m Channel map of the coefficients to update
   */
  void updateCoef(AngularPosition pos, ChannelLayout layout, ChannelMap m);

  AmbisonicOrder _order;
  AmbisonicNorm _norm;
  ambCoefTable_t _coefficients;  // maps the channel map to a vector of encoding coefficients
};

enum class AmbisonicDecodeType { Basic, MaxRe, InPhase };

/**
 * Ambisonic decoder:
 * Implements decoder for the FIRST_ORDER only. And it supports any layout, at the condition it as the corresponding
 * coefficients loaded. For the moment only STEREO and 5.1 layouts have been tested.
 */
class VS_EXPORT AmbDecoder : public AudioObject {
 public:
  /**
   * @brief Constructor.
   * @param outLayout layout of the output
   * @param coefficients table of decoding coefficients
   */
  explicit AmbDecoder(ChannelLayout l, const ambCoefTable_t &coef);

  /**
   * @brief Default destructor.
   */
  ~AmbDecoder();

  /**
   * @brief Process the decoding of the input to the output
   * @param out Output audio block
   * @param in Input audio block
   */
  void step(AudioBlock &out, const AudioBlock &in);
  void step(AudioBlock &inout);

  /**
   * @brief Sets table of decoding coefficients.
   * @param coefficients table of decoding coefficients
   */
  void setCoefficients(const ambCoefTable_t &coefficients);

 private:
  ChannelLayout _outLayout;
  ambCoefTable_t _coefficients;  // maps the channel map to a vector of decoding coefficients
};

/**
 * Ambisonic rotator:
 * Implements rotations for the FIRST_ORDER only.
 */
class VS_EXPORT AmbRotator : public AudioObject {
 public:
  /**
   * @brief Constructor.
   * @param outLayout layout of the output
   * @param coefficients table of decoding coefficients
   */
  explicit AmbRotator(const AmbisonicOrder o);

  /**
   * @brief Default destructor.
   */
  ~AmbRotator();

  /**
   * @brief Apply rotation
   * @param out Output audio block
   * @param in Input audio block
   */
  void step(AudioBlock &out, const AudioBlock &in);
  void step(AudioBlock &inout);

  /**
   * @brief Sets rotations.
   * @param Yaw, pitch, and roll of audio sound field.
   */
  void setRotation(double yaw, double pitch, double roll);

  /**
   * @brief Sets an offset to apply on the rotation.
   * @param Yaw, pitch, and roll of audio sound field.
   */
  void setRotationOffset(double yaw, double pitch, double roll);

  /**
   * @brief Gets rotations.
   * @param Yaw, pitch, and roll of audio sound field.
   */
  Vector3<double> getRotation() {
    std::lock_guard<std::mutex> lk(_rotMutex);
    return _rotation;
  }

  /**
   * @brief Gets offset rotation.
   * @param Yaw, pitch, and roll of audio sound field.
   */
  Vector3<double> getRotationOffset() {
    std::lock_guard<std::mutex> lk(_rotMutex);
    return _offset;
  }

 private:
  enum { YAW = 0, PITCH, ROLL };
  AmbisonicOrder _order;
  std::mutex _rotMutex;
  Vector3<double> _rotation;  // yaw, pitch, roll
  Vector3<double> _offset;    // yaw, pitch, roll
};

}  // namespace Audio
}  // namespace VideoStitch
