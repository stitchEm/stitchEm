// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputfile.hpp"

#include "liveaudio.hpp"
#include "guiconstants.hpp"
#include "liveprojectdefinition.hpp"
#include "encodedoutputhelpers.hpp"

#include "configurations/configurationoutputhdd.hpp"

#include "libvideostitch-gui/utils/bitratemodeenum.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/output.hpp"

#include <QDateTime>
#include <memory>

LiveOutputFile::LiveOutputFile(const VideoStitch::Ptv::Value* config,
                               const VideoStitch::Core::PanoDefinition* panoDefinition,
                               const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveWriterFactory(type),
      filename(getDefaultOutputFileName()),
      downsampling(1),
      codec(VideoStitch::VideoCodec::VideoCodecEnum::H264),
      h264Profile(DEFAULT_PROFILE),
      h264Level(),
      h264CommandLineArgs(),
      mjpegQualityScale(0),
      proresProfile(),
      bitrateMode(VBR),
      gop(-1),
      bframes(DEFAULT_B_FRAMES),
      bitrate(-1) {
  fillOutputValues(config, panoDefinition);
}

LiveOutputFile::~LiveOutputFile() {}

const QString LiveOutputFile::getIdentifier() const { return getFileName(); }

const QString LiveOutputFile::getOutputDisplayName() const { return getFileName(); }

int LiveOutputFile::getDownsamplingFactor() const { return downsampling; }

QString LiveOutputFile::getFileName() const { return filename; }

VideoStitch::VideoCodec::VideoCodecEnum LiveOutputFile::getCodec() const { return codec; }

int LiveOutputFile::getGOP() const { return gop; }

int LiveOutputFile::getBFrames() const { return bframes; }

int LiveOutputFile::getBitRate() const { return bitrate; }

QString LiveOutputFile::getH264Profile() const { return h264Profile; }

QString LiveOutputFile::getH264Level() const { return h264Level; }

int LiveOutputFile::getMjpegQualityScale() const { return mjpegQualityScale; }

QString LiveOutputFile::getProresProfile() const { return proresProfile; }

BitRateModeEnum LiveOutputFile::getBitRateMode() const { return bitrateMode; }

QList<VideoStitch::Audio::SamplingDepth> LiveOutputFile::getSupportedSamplingDepths(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const {
  // All the options have been checked and only these ones are compatible
  QList<VideoStitch::Audio::SamplingDepth> samplingDepths;
  switch (audioCodec) {
    case VideoStitch::AudioHelpers::AudioCodecEnum::MP3:
      samplingDepths << VideoStitch::Audio::SamplingDepth::INT16_P << VideoStitch::Audio::SamplingDepth::INT32_P
                     << VideoStitch::Audio::SamplingDepth::FLT_P;
      break;
    case VideoStitch::AudioHelpers::AudioCodecEnum::AAC:
    default:
      samplingDepths << VideoStitch::Audio::SamplingDepth::FLT_P;
      break;
  }
  return samplingDepths;
}

LiveOutputFactory::PanoSizeChange LiveOutputFile::supportPanoSizeChange(int newWidth, int newHeight) const {
  if (downsampling != 1) {
    return PanoSizeChange::SupportedWithUpdate;
  }

  if ((codec == VideoStitch::VideoCodec::VideoCodecEnum::H264) ||
      (codec == VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264)) {
    // Downsampling factor will be set to 1
    auto newPixelRate = VideoStitch::getPixelRate(newWidth, newHeight);
    QString newLevel = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(newPixelRate), codec);
    int newBitrate = VideoStitch::getDefaultBitrate(newPixelRate);
    if (newLevel != h264Level || newBitrate != bitrate) {
      return PanoSizeChange::SupportedWithUpdate;
    }
  }
  return PanoSizeChange::Supported;
}

QString LiveOutputFile::getPanoSizeChangeDescription(int newWidth, int newHeight) const {
  QString desc = QString("%0 (%1)\n").arg(getOutputTypeDisplayName()).arg(getOutputDisplayName()) +
                 tr("- Output size: %0x%1").arg(newWidth).arg(newHeight);

  if ((codec == VideoStitch::VideoCodec::VideoCodecEnum::H264) ||
      (codec == VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264)) {
    // Downsampling factor will be set to 1
    auto newPixelRate = VideoStitch::getPixelRate(newWidth, newHeight);
    QString newLevel = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(newPixelRate), codec);
    int newBitrate = VideoStitch::getDefaultBitrate(newPixelRate);
    desc += tr("\n- Level: %0\n- Max bitrate: %1 kbits/s").arg(newLevel).arg(newBitrate);
  }
  return desc;
}

void LiveOutputFile::updateForPanoSizeChange(int newWidth, int newHeight) {
  downsampling = 1;
  if ((codec == VideoStitch::VideoCodec::VideoCodecEnum::H264) ||
      (codec == VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264)) {
    auto newPixelRate = VideoStitch::getPixelRate(newWidth, newHeight);
    QString newLevel = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(newPixelRate), codec);
    int newBitrate = VideoStitch::getDefaultBitrate(newPixelRate);
    h264Level = newLevel;
    bitrate = newBitrate;
  }
}

void LiveOutputFile::setDownsamplingFactor(const unsigned int dsValue) { downsampling = dsValue; }

void LiveOutputFile::setFileName(const QString fileValue) {
  filename = fileValue;
  emit outputDisplayNameChanged(filename);
}

void LiveOutputFile::setGOP(unsigned int gopValue) { gop = gopValue; }

void LiveOutputFile::setBFrames(const unsigned int framesValue) { bframes = framesValue; }

void LiveOutputFile::setBitRate(const unsigned int bitrateValue) { bitrate = bitrateValue; }

void LiveOutputFile::setCodec(const VideoStitch::VideoCodec::VideoCodecEnum& codecValue) { codec = codecValue; }

void LiveOutputFile::setType(const QString typeValue) {
  type = VideoStitch::OutputFormat::getEnumFromString(typeValue);
}

void LiveOutputFile::setH264Profile(QString newH264Profile) { h264Profile = newH264Profile; }

void LiveOutputFile::setH264Level(QString newH264Level) { h264Level = newH264Level; }

void LiveOutputFile::setMjpegQualityScale(int newMjpegQualityScale) { mjpegQualityScale = newMjpegQualityScale; }

void LiveOutputFile::setProresProfile(QString newProresProfile) { proresProfile = newProresProfile; }

void LiveOutputFile::setBitRateMode(BitRateModeEnum newBitrateMode) { bitrateMode = newBitrateMode; }

void LiveOutputFile::fillOutputValues(const VideoStitch::Ptv::Value* config,
                                      const VideoStitch::Core::PanoDefinition* panoDefinition) {
  std::string fileName;
  VideoStitch::Parse::populateString("Ptv", *config, "filename", fileName, true);
  setFileName(QString::fromStdString(fileName));

  std::string extension;
  if (VideoStitch::Parse::populateString("Ptv", *config, "type", extension, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    type = VideoStitch::OutputFormat::getEnumFromString(QString(extension.c_str()));
  }

  int resizedWidth = DEFAULT_PANO_WIDTH;
  int resizedHeight = DEFAULT_PANO_HEIGHT;
  if (panoDefinition) {
    resizedWidth = int(panoDefinition->getWidth()) / downsampling;
    resizedHeight = int(panoDefinition->getHeight()) / downsampling;
  }

  std::string cod;
  if (VideoStitch::Parse::populateString("Ptv", *config, "video_codec", cod, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    codec = VideoStitch::VideoCodec::getEnumFromString(QString::fromStdString(cod));
  }
  switch (codec) {
    case VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264:
    case VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264:
    case VideoStitch::VideoCodec::VideoCodecEnum::H264: {
      auto pixelRate = VideoStitch::getPixelRate(resizedWidth, resizedHeight);
      std::string stdProfile, stdLevel, commandLine;
      if (VideoStitch::Parse::populateString("Ptv", *config, "profile", stdProfile, false) ==
          VideoStitch::Parse::PopulateResult_Ok) {
        h264Profile = QString::fromStdString(stdProfile);
      }
      if (VideoStitch::Parse::populateString("Ptv", *config, "level", stdLevel, false) ==
          VideoStitch::Parse::PopulateResult_Ok) {
        h264Level = QString::fromStdString(stdLevel);
      } else {
        h264Level = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(pixelRate), codec);
      }
      if (VideoStitch::Parse::populateString("Ptv", *config, "codec_command_line", commandLine, false) ==
          VideoStitch::Parse::PopulateResult_Ok) {
        h264CommandLineArgs = QString::fromStdString(commandLine);
      }
      if (VideoStitch::Parse::populateInt("Ptv", *config, "bitrate", bitrate, true) ==
          VideoStitch::Parse::PopulateResult_Ok) {
        bitrate /= TO_BITS;
      } else {
        bitrate = VideoStitch::getDefaultBitrate(pixelRate);
      }
      VideoStitch::Parse::populateInt("Ptv", *config, "gop", gop, true);
      break;
    }
    case VideoStitch::VideoCodec::VideoCodecEnum::MJPEG:
      VideoStitch::Parse::populateInt("Ptv", *config, "scale", mjpegQualityScale, false);
      break;
    case VideoStitch::VideoCodec::VideoCodecEnum::PRORES: {
      std::string stdProfile;
      if (VideoStitch::Parse::populateString("Ptv", *config, "profile", stdProfile, false) ==
          VideoStitch::Parse::PopulateResult_Ok) {
        proresProfile = QString::fromStdString(stdProfile);
      }
      break;
    }
    default:
      break;
  }

  std::string brMode;
  if (VideoStitch::Parse::populateString("Ptv", *config, "bitrate_mode", brMode, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    bitrateMode = BitRateModeEnum::getEnumFromDescriptor(QString::fromStdString(brMode));
  }

  if (VideoStitch::Parse::populateInt("Ptv", *config, "downsampling_factor", downsampling, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    downsampling = DEFAULT_DOWNSAMPLING;
  }

  if (VideoStitch::Parse::populateInt("Ptv", *config, "b_frames", bframes, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    bframes = DEFAULT_B_FRAMES;
  }
}

VideoStitch::Ptv::Value* LiveOutputFile::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = VideoStitch::OutputFormat::getStringFromEnum(type).toStdString();
  value->get("filename")->asString() = filename.toStdString();
  value->get("bitrate_mode")->asString() = bitrateMode.getDescriptor().toStdString();
  value->get("video_codec")->asString() = VideoStitch::VideoCodec::getStringFromEnum(codec).toStdString();
  value->get("b_frames")->asInt() = bframes;
  switch (codec) {
    case VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264:
    case VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264:
    case VideoStitch::VideoCodec::VideoCodecEnum::H264:
      value->get("profile")->asString() = h264Profile.toStdString();
      value->get("level")->asString() = h264Level.toStdString();
      if (!h264CommandLineArgs.isEmpty()) {
        value->get("codec_command_line")->asString() = h264CommandLineArgs.toStdString();
      }
      value->get("bitrate")->asInt() = bitrate * TO_BITS;
      value->get("gop")->asInt() = gop;
      break;
    case VideoStitch::VideoCodec::VideoCodecEnum::MJPEG:
      value->get("scale")->asInt() = mjpegQualityScale;
      break;
    case VideoStitch::VideoCodec::VideoCodecEnum::PRORES:
      value->get("profile")->asString() = proresProfile.toStdString();
      break;
    default:
      break;
  }

  if (hasLog) {
    const QString logFile = initializeAvStatsLogFile(VideoStitch::OutputFormat::getStringFromEnum(type));
    value->get("stats_log_file")->asString() = logFile.toStdString();
  }

  if (downsampling > 1) {
    value->get("downsampling_factor")->asInt() = downsampling;
  }
  audioConfig->serializeIn(value);
  return value;
}

QWidget* LiveOutputFile::createStatusWidget(QWidget* const parent) { return createStatusIcon(parent); }

QPixmap LiveOutputFile::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/save-hdd.png"); }

VideoStitch::Potential<VideoStitch::Output::Output> LiveOutputFile::createWriter(LiveProjectDefinition* project,
                                                                                 VideoStitch::FrameRate framerate) {
  std::unique_ptr<VideoStitch::Ptv::Value> save_hdd_parameters(serialize());
  save_hdd_parameters->get("filename")->asString() =
      save_hdd_parameters->get("filename")->asString() + "_" +
      QDateTime::currentDateTime().toString("yyyy-MM-dd-hh-mm-ss").toStdString();

  return VideoStitch::Output::create(
      *save_hdd_parameters, getIdentifier().toStdString(), project->getPanoConst()->getWidth(),
      project->getPanoConst()->getHeight(), framerate,
      VideoStitch::Audio::getSamplingRateFromInt(audioConfig->getSamplingRate()),
      VideoStitch::Audio::getSamplingDepthFromString(audioConfig->getSamplingFormat().toStdString().c_str()),
      VideoStitch::Audio::getChannelLayoutFromString(audioConfig->getChannelLayout().toStdString().c_str()));
}

OutputConfigurationWidget* LiveOutputFile::createConfigurationWidget(QWidget* const parent) {
  return new ConfigurationOutputHDD(this, type, parent);
}

QString LiveOutputFile::initializeAvStatsLogFile(QString name) {
  // We specify a writable location for the stats log file (cf VSA-3471 for more details)
  // in order to launch Vahana VR without administrator rights
  QString dir = QStandardPaths::writableLocation(QStandardPaths::DataLocation);
  // libAV will add a prefix to the file name, so we can't escape it and it's forbidden to have space in the path
  dir.replace(" ", "_");
  // libAv expect an existing directory
  QDir().mkpath(dir);
  QString logFile = QString("%0/%1.log").arg(dir).arg(name);
  return logFile;
}
