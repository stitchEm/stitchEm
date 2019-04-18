// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>
#include <QApplication>

namespace VideoStitch {
namespace OutputFormat {

/**
 * @brief The available output formats. (Vahana and Studio)
 */
enum class OutputFormatEnum {
  MP4,
  MOV,
  JPG,
  PNG,
  PPM,
  PAM,
  RAW,
  TIF,
  YUV420P,
  RTMP,
  YOUTUBE,
  DECKLINK,
  AJA,
  OCULUS,
  STEAMVR,
  CUSTOM,
  UNKNOWN
};

/**
 * @brief Gets the translated string for the UI.
 * @param value The codec.
 * @return The UI string.
 */
static inline QString getDisplayNameFromEnum(const OutputFormatEnum& value) {
  switch (value) {
    case OutputFormatEnum::MP4:
      return QApplication::translate("OutputFormat", "MP4");
    case OutputFormatEnum::MOV:
      return QApplication::translate("OutputFormat", "MOV");
    case OutputFormatEnum::JPG:
      return QApplication::translate("OutputFormat", "JPEG sequence");
    case OutputFormatEnum::PNG:
      return QApplication::translate("OutputFormat", "PNG sequence");
    case OutputFormatEnum::PPM:
      return QApplication::translate("OutputFormat", "PPM sequence");
    case OutputFormatEnum::PAM:
      return QApplication::translate("OutputFormat", "PAM sequence");
    case OutputFormatEnum::RAW:
      return QApplication::translate("OutputFormat", "Raw sequence");
    case OutputFormatEnum::TIF:
      return QApplication::translate("OutputFormat", "TIFF sequence");
    case OutputFormatEnum::YUV420P:
      return QApplication::translate("OutputFormat", "YUV420p");
    case OutputFormatEnum::RTMP:
      return QApplication::translate("OutputFormat", "RTMP");
    case OutputFormatEnum::YOUTUBE:
      return QApplication::translate("OutputFormat", "YouTube");
    case OutputFormatEnum::DECKLINK:
      return QApplication::translate("OutputFormat", "Decklink");
    case OutputFormatEnum::OCULUS:
      return QApplication::translate("OutputFormat", "Oculus");
    case OutputFormatEnum::STEAMVR:
      return QApplication::translate("OutputFormat", "Steam VR");
    case OutputFormatEnum::AJA:
      return QApplication::translate("OutputFormat", "AJA");
    case OutputFormatEnum::CUSTOM:
      return QApplication::translate("OutputFormat", "Custom");
    default:
      return QString();
  }
}

/**
 * @brief Gets the string prepared for configuration.
 * @param value The codec.
 * @return The configuration string.
 */
static inline QString getStringFromEnum(const OutputFormatEnum& value) {
  switch (value) {
    case OutputFormatEnum::MP4:
      return QStringLiteral("mp4");
    case OutputFormatEnum::MOV:
      return QStringLiteral("mov");
    case OutputFormatEnum::JPG:
      return QStringLiteral("jpg");
    case OutputFormatEnum::PNG:
      return QStringLiteral("png");
    case OutputFormatEnum::PPM:
      return QStringLiteral("ppm");
    case OutputFormatEnum::PAM:
      return QStringLiteral("pam");
    case OutputFormatEnum::RAW:
      return QStringLiteral("raw");
    case OutputFormatEnum::TIF:
      return QStringLiteral("tif");
    case OutputFormatEnum::YUV420P:
      return QStringLiteral("yuv420p");
    case OutputFormatEnum::RTMP:
      return QStringLiteral("rtmp");
    case OutputFormatEnum::YOUTUBE:
      return QStringLiteral("youtube");
    case OutputFormatEnum::DECKLINK:
      return QStringLiteral("decklink");
    case OutputFormatEnum::OCULUS:
      return QStringLiteral("oculus");
    case OutputFormatEnum::STEAMVR:
      return QStringLiteral("steamvr");
    case OutputFormatEnum::AJA:
      return QStringLiteral("aja");
    case OutputFormatEnum::CUSTOM:
      return QStringLiteral("custom");
    default:
      return QString();
  }
}

/**
 * @brief Gets the enum value from a configuration string.
 * @param value The configuration string.
 * @return The enumerator.
 */
static inline OutputFormatEnum getEnumFromString(const QString value) {
  if (value == "mp4") {
    return OutputFormatEnum::MP4;
  } else if (value == "mov") {
    return OutputFormatEnum::MOV;
  } else if (value == "jpg") {
    return OutputFormatEnum::JPG;
  } else if (value == "png") {
    return OutputFormatEnum::PNG;
  } else if (value == "ppm") {
    return OutputFormatEnum::PPM;
  } else if (value == "pam") {
    return OutputFormatEnum::PAM;
  } else if (value == "raw") {
    return OutputFormatEnum::RAW;
  } else if (value == "tif") {
    return OutputFormatEnum::TIF;
  } else if (value == "yuv420p") {
    return OutputFormatEnum::YUV420P;
  } else if (value == "rtmp") {
    return OutputFormatEnum::RTMP;
  } else if (value == "youtube") {
    return OutputFormatEnum::YOUTUBE;
  } else if (value == "decklink") {
    return OutputFormatEnum::DECKLINK;
  } else if (value == "oculus") {
    return OutputFormatEnum::OCULUS;
  } else if (value == "steamvr") {
    return OutputFormatEnum::STEAMVR;
  } else if (value == "aja") {
    return OutputFormatEnum::AJA;
  } else if (value == "custom") {
    return OutputFormatEnum::CUSTOM;
  } else {
    return OutputFormatEnum::UNKNOWN;
  }
}

/**
 * @brief Determines if a format is a file.
 * @param value The output format value.
 * @return True if it's a file.
 */
static inline bool isFileFormat(const OutputFormatEnum& value) {
  return value == OutputFormatEnum::MP4 || value == OutputFormatEnum::MOV || value == OutputFormatEnum::JPG ||
         value == OutputFormatEnum::PNG || value == OutputFormatEnum::PPM || value == OutputFormatEnum::PAM ||
         value == OutputFormatEnum::RAW || value == OutputFormatEnum::TIF || value == OutputFormatEnum::YUV420P;
}

/**
 * @brief Determines if a format is a video.
 * @param value The output format value.
 * @return True if it's a video.
 */
static inline bool isVideoFormat(const OutputFormatEnum& value) {
  return value == OutputFormatEnum::MP4 || value == OutputFormatEnum::MOV;
}

/**
 * @brief Determines if a format is an image.
 * @param value The output format value.
 * @return True if it's an image.
 */
static inline bool isImageFormat(const OutputFormatEnum& value) {
  return value == OutputFormatEnum::JPG || value == OutputFormatEnum::PNG || value == OutputFormatEnum::PPM ||
         value == OutputFormatEnum::PAM || value == OutputFormatEnum::RAW || value == OutputFormatEnum::TIF;
}

/**
 * @brief Gets the supported video formats.
 * @return A list of video formats.
 */
static inline QList<OutputFormatEnum> getSupportedVideoFormats() {
  static QList<OutputFormatEnum> supportedVideoFormats;
  if (supportedVideoFormats.isEmpty()) {
    supportedVideoFormats.append(OutputFormatEnum::MP4);
    supportedVideoFormats.append(OutputFormatEnum::MOV);
  }
  return supportedVideoFormats;
}

/**
 * @brief Gets the supported image formats.
 * @return A list of image formats.
 */
static inline QList<OutputFormatEnum> getSupportedImageFormats() {
  static QList<OutputFormatEnum> supportImageFormats;
  if (supportImageFormats.isEmpty()) {
    supportImageFormats.append(OutputFormatEnum::JPG);
    supportImageFormats.append(OutputFormatEnum::PNG);
    supportImageFormats.append(OutputFormatEnum::TIF);
    // PPM and PAM don't work for the moment
  }
  return supportImageFormats;
}
}  // namespace OutputFormat
}  // namespace VideoStitch
