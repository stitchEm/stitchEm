// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videocodecs.hpp"
#include <QApplication>

namespace VideoStitch {
namespace VideoCodec {

VideoCodecEnum getEnumFromString(const QString& codec) {
  for (auto it : VideoEncoderEnum::getDescriptorsList()) {
    if (it.at(1) == codec) {
      return VideoEncoderEnum::getEnumFromDescriptor(it).getValue();
    }
  }
  return VideoCodecEnum::UNKNOWN;
}

void VideoCodecClass::initDescriptions(QMap<Enum, QStringList>& enumToString) {
  enumToString[VideoCodecEnum::MPEG2] =
      QStringList({QApplication::translate("VideoCodec", "MPEG2"), QStringLiteral("mpeg2")});
  enumToString[VideoCodecEnum::MPEG4] =
      QStringList({QApplication::translate("VideoCodec", "MPEG4"), QStringLiteral("mpeg4")});
  enumToString[VideoCodecEnum::H264] =
      QStringList({QApplication::translate("VideoCodec", "H264"), QStringLiteral("h264")});
  enumToString[VideoCodecEnum::HEVC] =
      QStringList({QApplication::translate("VideoCodec", "HEVC"), QStringLiteral("hevc")});
  enumToString[VideoCodecEnum::MJPEG] =
      QStringList({QApplication::translate("VideoCodec", "Motion JPEG"), QStringLiteral("mjpeg")});
  enumToString[VideoCodecEnum::PRORES] =
      QStringList({QApplication::translate("VideoCodec", "ProRes"), QStringLiteral("prores")});
  enumToString[VideoCodecEnum::QUICKSYNC_H264] =
      QStringList({QApplication::translate("VideoCodec", "H264 (Intel QuickSync)"), QStringLiteral("h264_qsv")});
  enumToString[VideoCodecEnum::NVENC_H264] =
      QStringList({QApplication::translate("VideoCodec", "H264 (Nvidia NVENC)"), QStringLiteral("h264_nvenc")});
  enumToString[VideoCodecEnum::NVENC_HEVC] =
      QStringList({QApplication::translate("VideoCodec", "HEVC (Nvidia NVENC)"), QStringLiteral("hevc_nvenc")});
  enumToString[VideoCodecEnum::UNKNOWN] =
      QStringList({QApplication::translate("VideoCodec", "Unknown"), QStringLiteral("unknown")});
  // additionals descriptors
  VideoEncoderEnum::descriptorToEnum[QStringList(
      {QApplication::translate("VideoCodec", "H264"), QStringLiteral("h264_x264")})] = VideoCodecEnum::H264;
  VideoEncoderEnum::descriptorToEnum[QStringList(
      {QApplication::translate("VideoCodec", "H264"), QStringLiteral("x264")})] = VideoCodecEnum::H264;
  VideoEncoderEnum::descriptorToEnum[QStringList({QApplication::translate("VideoCodec", "H264 (Intel QuickSync)"),
                                                  QStringLiteral("qsv")})] = VideoCodecEnum::QUICKSYNC_H264;
  VideoEncoderEnum::descriptorToEnum[QStringList({QApplication::translate("VideoCodec", "H264 (Intel QuickSync)"),
                                                  QStringLiteral("qsv_h264")})] = VideoCodecEnum::QUICKSYNC_H264;
  VideoEncoderEnum::descriptorToEnum[QStringList({QApplication::translate("VideoCodec", "H264 (Nvidia NVENC)"),
                                                  QStringLiteral("nvenc")})] = VideoCodecEnum::NVENC_H264;
  VideoEncoderEnum::descriptorToEnum[QStringList({QApplication::translate("VideoCodec", "H264 (Nvidia NVENC)"),
                                                  QStringLiteral("nvenc_h264")})] = VideoCodecEnum::NVENC_H264;
  VideoEncoderEnum::descriptorToEnum[QStringList({QApplication::translate("VideoCodec", "HEVC (Nvidia NVENC)"),
                                                  QStringLiteral("nvenc_hevc")})] = VideoCodecEnum::NVENC_HEVC;
}

const VideoCodecClass::Enum VideoCodecClass::defaultValue = VideoCodecEnum::UNKNOWN;

}  // namespace VideoCodec
}  // namespace VideoStitch
