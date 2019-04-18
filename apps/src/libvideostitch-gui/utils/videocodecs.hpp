// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>
#include <QApplication>
#include <QList>
#include "outputformat.hpp"
#include "smartenum.hpp"
#include <libvideostitch/gpu_device.hpp>

namespace VideoStitch {
namespace VideoCodec {

/**
 * @brief The available video codecs.
 */
enum class VideoCodecEnum { MPEG2, MPEG4, H264, HEVC, MJPEG, PRORES, QUICKSYNC_H264, NVENC_H264, NVENC_HEVC, UNKNOWN };

class VS_GUI_EXPORT VideoCodecClass {
 public:
  typedef VideoCodecEnum Enum;

  static void initDescriptions(QMap<Enum, QStringList>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<VideoCodecClass, QStringList> VideoEncoderEnum;

/**
 * @brief Gets the translated string for the UI.
 * @param value The codec.
 * @return The UI string.
 */
VS_GUI_EXPORT VideoCodecEnum getEnumFromString(const QString& codec);

static inline VideoCodecEnum getValueFromDescriptor(const QStringList& codec) {
  return VideoEncoderEnum::getValueFromDescriptor(codec);
}

/**
 * @brief Gets the string prepared for configuration.
 * @param value The codec.
 * @return The configuration string.
 */
static inline QString getDisplayNameFromEnum(const VideoCodecEnum& value) {
  return VideoEncoderEnum::getDescriptorFromEnum(value).at(0);
}

/**
 * @brief Gets the enum value from a configuration string.
 * @param value The configuration string.
 * @return The enumerator.
 */
static inline QString getStringFromEnum(const VideoCodecEnum& value) {
  return VideoEncoderEnum::getDescriptorFromEnum(value).at(1);
}

/**
 * @brief Gets the list of the supported codecs for a given output video format.
 * @param value The output video format.
 * @return A list of zero or more video codecs.
 */
static inline QList<VideoCodecEnum> getSupportedCodecsFor(const OutputFormat::OutputFormatEnum& value) {
  QList<VideoCodecEnum> supportedCodecs;
  if (value == OutputFormat::OutputFormatEnum::MP4) {
    supportedCodecs.append(VideoCodecEnum::H264);
    supportedCodecs.append(VideoCodecEnum::NVENC_H264);
    supportedCodecs.append(VideoCodecEnum::NVENC_HEVC);
    supportedCodecs.append(VideoCodecEnum::MPEG4);
    supportedCodecs.append(VideoCodecEnum::MPEG2);
    supportedCodecs.append(VideoCodecEnum::MJPEG);
  } else if (value == OutputFormat::OutputFormatEnum::MOV) {
    supportedCodecs.append(VideoCodecEnum::H264);
    supportedCodecs.append(VideoCodecEnum::NVENC_H264);
    supportedCodecs.append(VideoCodecEnum::NVENC_HEVC);
    supportedCodecs.append(VideoCodecEnum::MPEG4);
    supportedCodecs.append(VideoCodecEnum::MPEG2);
    supportedCodecs.append(VideoCodecEnum::MJPEG);
    supportedCodecs.append(VideoCodecEnum::PRORES);
  }
  return supportedCodecs;
}
}  // namespace VideoCodec
}  // namespace VideoStitch
