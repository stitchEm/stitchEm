// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "encodedoutputhelpers.hpp"

#include "libvideostitch-gui/utils/h264settingsenum.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include "libvideostitch/frame.hpp"

namespace VideoStitch {

int getDefaultBitrate(double pixelRate) {
  // Stéphane Valente:
  // >> For the default bitrate, let's assume that 1920x1080 @ 30 fps, 6 Mbit/s is of reasonable quality
  // >> Using the infamous rule of the thumb "each time your pixelrate quadruples, your bitrate needs to double",
  // >> you can have a reasonable quality bitrate by doing:
  return int(6000.0 * std::sqrt(pixelRate / (1920.0 * 1080.0 * 30.0))) + 1;
}

QString getLevelFromMacroblocksRate(int macroblocksRate, VideoCodec::VideoCodecEnum codec) {
  // https://en.wikipedia.org/wiki/H.264/MPEG-4_AVC#Levels
  static const QMap<int, QString> levelByMacroblocksRate[2] = {
      QMap<int, QString>{{40500, "3"},
                         {108000, "3.1"},
                         {216000, "3.2"},
                         {245760, "4"},
                         {245760, "4.1"},
                         {522240, "4.2"},
                         {589824, "5"},
                         {983040, "5.1"},
                         {2073600, "5.2"},
                         {4177920, "6"},
                         {8355840, "6.1"},
                         {16711680, "6.2"}},
      // https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding
      QMap<int, QString>{{64800, "3"},
                         {129600, "3.1"},
                         {261120, "4"},
                         {522240, "4.1"},
                         {1044480, "5"},
                         {2088960, "5.1"},
                         {4177920, "5.2"},
                         {4177920, "6"},
                         {8355840, "6.1"},
                         {16711680, "6.2"}}};

  int cod = (codec != VideoCodec::VideoCodecEnum::H264) && (codec != VideoCodec::VideoCodecEnum::QUICKSYNC_H264) &&
            (codec != VideoCodec::VideoCodecEnum::NVENC_H264);
  auto it = levelByMacroblocksRate[cod].lowerBound(macroblocksRate);
  if (it != levelByMacroblocksRate[cod].constEnd()) {
    return it.value();
  } else {
    return QString();
  }
}

int getMaxConstantBitRate(QString profile, QString level, VideoCodec::VideoCodecEnum codec) {
  // https://en.wikipedia.org/wiki/H.264/MPEG-4_AVC#Levels
  static const QHash<QString, int> maxMainBitrateByLevel[2] = {
      QHash<QString, int>{{"3", 10000},
                          {"3.1", 14000},
                          {"3.2", 20000},
                          {"4", 20000},
                          {"4.1", 50000},
                          {"4.2", 50000},
                          {"5", 135000},
                          {"5.1", 240000},
                          {"5.2", 240000},
                          {"6", 240000},
                          {"6.1", 480000},
                          {"6.2", 800000}},
      // https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding
      QHash<QString, int>{{"3", 6000},
                          {"3.1", 10000},
                          {"3.2", 10000},
                          {"4", 12000},
                          {"4.1", 20000},
                          {"4.2", 20000},
                          {"5", 25000},
                          {"5.1", 40000},
                          {"5.2", 60000},
                          {"6", 60000},
                          {"6.1", 120000},
                          {"6.2", 240000}}};

  int cod = (codec == VideoCodec::VideoCodecEnum::HEVC) || (codec == VideoCodec::VideoCodecEnum::NVENC_HEVC);
  int maxBitrate = maxMainBitrateByLevel[cod].value(level, 0);
  if (profile == H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::HIGH)) {
    maxBitrate = std::round(maxBitrate * 1.25);
  } else if (profile == H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::HIGH10)) {
    maxBitrate *= 3;
  } else if (profile == H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::HIGH422) ||
             profile == H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::HIGH444)) {
    maxBitrate *= 4;
  }
  return maxBitrate;
}

int getMacroblocksRate(double pixelRate) { return int(pixelRate / 256.0) + 1; }

double getPixelRate(int width, int height) {
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  double fps = double(frameRate.num) / double(frameRate.den);
  return fps * double(width) * double(height);
}

}  // namespace VideoStitch
