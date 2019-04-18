// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>
#include <QCoreApplication>

namespace VideoStitch {
namespace InputFormat {

static const QString& VIDEO_FORMATS("(*.mp4 *.mov *.mpg *.avi *.mkv *.mp2 *.3gp *.m4v *.mpeg *.ogv *.ogg *.wmv)");
static const QString& IMAGE_FORMATS("(*.jpeg *.jpg *.png)");

/**
 * @brief The available input formats. (Vahana and Studio)
 */
enum class InputFormatEnum {
  PROCEDURAL,
  MEDIA,
  MAGEWELL,
  MAGEWELLPRO,
  DECKLINK,
  XIMEA,
  PORTAUDIO,
  AUDIOPROCEDURAL,
  AJA,
  NETWORK,
  V4L2,
  INVALID
};

/**
 * @brief Gets the translated string for the UI.
 * @param value The codec.
 * @return The UI string.
 */
static inline QString getDisplayNameFromEnum(const InputFormatEnum& value) {
  switch (value) {
    case InputFormatEnum::PROCEDURAL:
      return QCoreApplication::translate("InputFormat", "Test inputs");
    case InputFormatEnum::MEDIA:
      return QCoreApplication::translate("InputFormat", "Media files");
    case InputFormatEnum::MAGEWELL:
      return QCoreApplication::translate("InputFormat", "Magewell");
    case InputFormatEnum::MAGEWELLPRO:
      return QCoreApplication::translate("InputFormat", "Magewell Pro");
    case InputFormatEnum::DECKLINK:
      return QCoreApplication::translate("InputFormat", "Blackmagic DeckLink");
    case InputFormatEnum::XIMEA:
      return QCoreApplication::translate("InputFormat", "Ximea (BETA)");
    case InputFormatEnum::PORTAUDIO:
      return QCoreApplication::translate("InputFormat", "Portable Audio IO");
    case InputFormatEnum::AUDIOPROCEDURAL:
      return QCoreApplication::translate("InputFormat", "Audio Test input");
    case InputFormatEnum::AJA:
      return QCoreApplication::translate("InputFormat", "AJA (BETA)");
    case InputFormatEnum::NETWORK:
      return QCoreApplication::translate("InputFormat", "RTSP inputs");
    case InputFormatEnum::V4L2:
      return QCoreApplication::translate("InputFormat", "Video for Linux Two");
    case InputFormatEnum::INVALID:
      return QCoreApplication::translate("InputFormat", "Invalid");
    default:
      return QStringLiteral("");
  }
}

/**
 * @brief Gets the string prepared for configuration.
 * @param value The codec.
 * @return The configuration string.
 */
static inline QString getStringFromEnum(const InputFormatEnum& value) {
  switch (value) {
    case InputFormatEnum::PROCEDURAL:
      return QStringLiteral("procedural");
    case InputFormatEnum::MEDIA:
      return QStringLiteral("media");
    case InputFormatEnum::MAGEWELL:
      return QStringLiteral("magewell");
    case InputFormatEnum::MAGEWELLPRO:
      return QStringLiteral("magewellpro");
    case InputFormatEnum::DECKLINK:
      return QStringLiteral("decklink");
    case InputFormatEnum::XIMEA:
      return QStringLiteral("ximea");
    case InputFormatEnum::PORTAUDIO:
      return QStringLiteral("portaudio");
    case InputFormatEnum::AUDIOPROCEDURAL:
      return QStringLiteral("audio procedural");
    case InputFormatEnum::AJA:
      return QStringLiteral("aja");
    case InputFormatEnum::NETWORK:
      return QStringLiteral("network");
    case InputFormatEnum::V4L2:
      return QStringLiteral("v4l2");
    case InputFormatEnum::INVALID:
      return QStringLiteral("invalid");
    default:
      return QStringLiteral("");
  }
}

/**
 * @brief Gets the enum value from a configuration string.
 * @param value The configuration string.
 * @return The enumerator.
 */
static inline InputFormatEnum getEnumFromString(const QString& value) {
  if (value == "procedural") {
    return InputFormatEnum::PROCEDURAL;
  } else if (value == "media") {
    return InputFormatEnum::MEDIA;
  } else if (value == "magewell") {
    return InputFormatEnum::MAGEWELL;
  } else if (value == "magewellpro") {
    return InputFormatEnum::MAGEWELLPRO;
  } else if (value == "decklink") {
    return InputFormatEnum::DECKLINK;
  } else if (value == "ximea") {
    return InputFormatEnum::XIMEA;
  } else if (value == "portaudio") {
    return InputFormatEnum::PORTAUDIO;
  } else if (value == "audio procedural") {
    return InputFormatEnum::AUDIOPROCEDURAL;
  } else if (value == "aja") {
    return InputFormatEnum::AJA;
  } else if (value == "network") {
    return InputFormatEnum::NETWORK;
  } else if (value == "v4l2") {
    return InputFormatEnum::V4L2;
  } else {
    return InputFormatEnum::INVALID;
  }
}

/**
 * @brief Determines if an input format is a video.
 * @param input The input format string.
 * @return True if it's a video input.
 */
static inline bool isVideoFile(const QString& input) {
  return input.endsWith(".mp4", Qt::CaseInsensitive) || input.endsWith(".mov", Qt::CaseInsensitive);
}

/**
 * @brief Determines if an input format is a stream.
 * @param input The input format string.
 * @return True if it's a stream input.
 */
static inline bool isVideoStream(const QString& input) {
  return input.startsWith("rtsp://", Qt::CaseInsensitive) || input.startsWith("rtmp://", Qt::CaseInsensitive);
}
}  // namespace InputFormat
}  // namespace VideoStitch
