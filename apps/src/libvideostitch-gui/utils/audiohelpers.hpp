// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>
#include <QApplication>
#include "libvideostitch/audio.hpp"

namespace VideoStitch {
namespace AudioHelpers {

static const unsigned DEFAULT_AUDIO_BITRATE(192);                   // kbps
static const std::vector<int> nbInputChannelsSupported({1, 2, 4});  // support only 1, 2 or 4 input channels
static const std::vector<Audio::SamplingRate> samplingRatesSupported(
    {Audio::SamplingRate::SR_44100, Audio::SamplingRate::SR_48000});  // support only 1, 2 or 4 input channels

/**
 * @brief The available audio codecs.
 */
enum class AudioCodecEnum { MP3, AAC, UNKNOWN };

/**
 * @brief Gets the translated string for the UI.
 * @param value The codec.
 * @return The UI string.
 */
static inline QString getDisplayNameFromCodec(const AudioCodecEnum& value) {
  if (value == AudioCodecEnum::MP3) {
    return QApplication::translate("AudioCodec", "MP3");
  } else {
    return QApplication::translate("AudioCodec", "AAC");
  }
}

/**
 * @brief Gets the string prepared for configuration.
 * @param value The codec.
 * @return The configuration string.
 */
static inline QString getStringFromCodec(const AudioCodecEnum& value) {
  if (value == AudioCodecEnum::MP3) {
    return QStringLiteral("mp3");
  } else {
    return QStringLiteral("aac");
  }
}

/**
 * @brief Gets the enum value from a configuration string.
 * @param value The configuration string.
 * @return The enumerator.
 */
static inline AudioCodecEnum getCodecFromString(const QString value) {
  if (value == "mp3") {
    return AudioCodecEnum::MP3;
  } else if (value == "aac") {
    return AudioCodecEnum::AAC;
  } else {
    return AudioCodecEnum::UNKNOWN;
  }
}

/**
 * @brief Gets the sampling rate of an audio codec.
 * @param codec The audio codec.
 * @return The default sampling rate.
 */
static inline int getDefaultSamplingRate(const AudioCodecEnum& codec) {
  if (codec == AudioCodecEnum::MP3) {
    return VideoStitch::Audio::getIntFromSamplingRate(VideoStitch::Audio::SamplingRate::SR_44100);
  } else {
    return VideoStitch::Audio::getIntFromSamplingRate(VideoStitch::Audio::SamplingRate::SR_48000);
  }
}

/**
 * @brief Converts the Sampling rate enumerator into a human readable string
 * @param rate Sampling rate enumerator
 * @return The string to be used in the UI
 */
static inline QString getSampleRateString(const Audio::SamplingRate rate) {
  const int value = Audio::getIntFromSamplingRate(rate);
  if (value == 0) {
    return QCoreApplication::translate("AudioHelpers", "No sampling");
  } else {
    return QCoreApplication::translate("AudioHelpers", "%0 Hz").arg(value);
  }
}

/**
 * @brief Returns the standard audio bitrates to be shown in the UI
 * @return A list of standard audio bitrates
 */
static inline QStringList getAudioBitrates() {
  // IBC-demo
  return QStringList() << "64"
                       << "96"
                       << "128"
                       << "192"
                       << "256"
                       << "512";
}

}  // namespace AudioHelpers
}  // namespace VideoStitch
