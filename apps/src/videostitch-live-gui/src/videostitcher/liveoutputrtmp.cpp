// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputrtmp.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/output.hpp"
#include "liveprojectdefinition.hpp"
#include "liveaudio.hpp"
#include "guiconstants.hpp"
#include "encodedoutputhelpers.hpp"

#include "configurations/configurationoutputstreaming.hpp"

#include "libvideostitch-gui/mainwindow/versionHelper.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/h264settingsenum.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include "plugin/pluginscontroller.hpp"

#include <QMovie>

#include <memory>

static const QString& LOADING_ICON(":/live/icons/assets/icon/live/live.gif");

LiveOutputRTMP::LiveOutputRTMP(const VideoStitch::Ptv::Value* config,
                               const VideoStitch::Core::PanoDefinition* panoDefinition,
                               const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveWriterFactory(type),
      url(DEFAULT_RTMP),
      downsampling(1),
      tune(),
      profile(DEFAULT_PROFILE),
      level(),
      gop(-1),
      bframes(DEFAULT_B_FRAMES),
      bitrate(-1),
      bufferSize(-1),
      memType(),
      bitrateMode(VBR),
      qualityBalance(DEFAULT_QUALITY_BALANCE),
      qp(-1),
      cbrPaddingEnabled(false),
      minBitrate(-1),
      maxBitrate(-1),
      statusWidget(nullptr) {
  fillOutputValues(config, panoDefinition);
}

LiveOutputRTMP::~LiveOutputRTMP() {}

const QString LiveOutputRTMP::getIdentifier() const { return getUrl(); }

const QString LiveOutputRTMP::getOutputDisplayName() const { return getUrl(); }

const BitRateModeEnum LiveOutputRTMP::getBitRateMode() const { return bitrateMode; }

int LiveOutputRTMP::getQualityBalance() const { return qualityBalance; }

int LiveOutputRTMP::getGOP() const { return gop; }

int LiveOutputRTMP::getBFrames() const { return bframes; }

int LiveOutputRTMP::getBitRate() const { return bitrate; }

int LiveOutputRTMP::getBufferSize() const { return bufferSize; }

int LiveOutputRTMP::getMinBitrate() const { return minBitrate; }

int LiveOutputRTMP::getTargetUsage() const { return targetUsage; }

QString LiveOutputRTMP::getPubUser() const { return pubUser; }
QString LiveOutputRTMP::getPubPasswd() const { return pubPassword; }

bool LiveOutputRTMP::showParameters() const { return true; }

bool LiveOutputRTMP::needsAuthentication() const { return !pubUser.isEmpty(); }

bool LiveOutputRTMP::forceConstantBitRate() const { return false; }

QStringList LiveOutputRTMP::getEncoder() const { return encoder; }

QList<QStringList> LiveOutputRTMP::getEncoderList() const { return encoderList; }

QString LiveOutputRTMP::getPreset() const { return preset; }

QString LiveOutputRTMP::getTune() const { return tune; }

QString LiveOutputRTMP::getProfile() const { return profile; }

QString LiveOutputRTMP::getLevel() const { return level; }

bool LiveOutputRTMP::cbrPaddingIsEnabled() const { return cbrPaddingEnabled; }

QList<VideoStitch::Audio::SamplingDepth> LiveOutputRTMP::getSupportedSamplingDepths(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const {
  QList<VideoStitch::Audio::SamplingDepth> samplingDepths;
  switch (audioCodec) {
    case VideoStitch::AudioHelpers::AudioCodecEnum::MP3:
      samplingDepths << VideoStitch::Audio::SamplingDepth::INT16 << VideoStitch::Audio::SamplingDepth::FLT
                     << VideoStitch::Audio::SamplingDepth::DBL << VideoStitch::Audio::SamplingDepth::INT16_P
                     << VideoStitch::Audio::SamplingDepth::INT32_P << VideoStitch::Audio::SamplingDepth::FLT_P
                     << VideoStitch::Audio::SamplingDepth::DBL_P;
      break;
    case VideoStitch::AudioHelpers::AudioCodecEnum::AAC:
    default:
      samplingDepths << VideoStitch::Audio::SamplingDepth::INT16 << VideoStitch::Audio::SamplingDepth::INT32
                     << VideoStitch::Audio::SamplingDepth::FLT;
      break;
  }
  return samplingDepths;
}

LiveOutputFactory::PanoSizeChange LiveOutputRTMP::supportPanoSizeChange(int newWidth, int newHeight) const {
  if (downsampling != 1) {
    return PanoSizeChange::SupportedWithUpdate;
  }

  // Downsampling factor will be set to 1
  auto newPixelRate = VideoStitch::getPixelRate(newWidth, newHeight);
  QString newLevel = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(newPixelRate),
                                                              VideoStitch::VideoCodec::getValueFromDescriptor(encoder));
  int newBitrate = VideoStitch::getDefaultBitrate(newPixelRate);
  if (newLevel != level || newBitrate != bitrate) {
    return PanoSizeChange::SupportedWithUpdate;
  }
  if (bufferSize != -1 && newBitrate != bufferSize) {
    return PanoSizeChange::SupportedWithUpdate;
  }
  return PanoSizeChange::Supported;
}

QString LiveOutputRTMP::getPanoSizeChangeDescription(int newWidth, int newHeight) const {
  QString desc = QString("%0 (%1)\n").arg(getOutputTypeDisplayName()).arg(getOutputDisplayName()) +
                 tr("- Output size: %0x%1").arg(newWidth).arg(newHeight);

  // Downsampling factor will be set to 1
  auto newPixelRate = VideoStitch::getPixelRate(newWidth, newHeight);
  QString newLevel = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(newPixelRate),
                                                              VideoStitch::VideoCodec::getValueFromDescriptor(encoder));
  int newBitrate = VideoStitch::getDefaultBitrate(newPixelRate);
  desc += tr("\n- Level: %0\n- Max bitrate: %1 kbits/s").arg(newLevel).arg(newBitrate);
  if (bufferSize != -1) {
    desc += tr("\n- Buffer size: %0 kbits").arg(newBitrate);
  }
  return desc;
}

void LiveOutputRTMP::updateForPanoSizeChange(int newWidth, int newHeight) {
  downsampling = 1;
  auto newPixelRate = VideoStitch::getPixelRate(newWidth, newHeight);
  QString newLevel = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(newPixelRate),
                                                              VideoStitch::VideoCodec::getValueFromDescriptor(encoder));
  int newBitrate = VideoStitch::getDefaultBitrate(newPixelRate);
  level = newLevel;
  bitrate = newBitrate;
  if (bufferSize != -1) {
    bufferSize = newBitrate;
  }
}

void LiveOutputRTMP::setUrl(const QString urlName) {
  url = urlName;
  emit outputDisplayNameChanged(url);
}

void LiveOutputRTMP::setEncoder(QStringList newEncoder) { encoder = newEncoder; }

void LiveOutputRTMP::setPreset(QString newPreset) { preset = newPreset; }

void LiveOutputRTMP::setTune(QString newTune) { tune = newTune; }

void LiveOutputRTMP::setProfile(QString newProfile) { profile = newProfile; }

void LiveOutputRTMP::setLevel(QString newLevel) { level = newLevel; }

void LiveOutputRTMP::setGOP(unsigned int gopValue) { gop = gopValue; }

void LiveOutputRTMP::setBitRateMode(const BitRateModeEnum& modeValue) { bitrateMode = modeValue; }

void LiveOutputRTMP::setQualityBalance(int newQualityBalance) { qualityBalance = newQualityBalance; }

void LiveOutputRTMP::setBFrames(const unsigned int framesValue) { bframes = framesValue; }

void LiveOutputRTMP::setBitRate(const unsigned int bitrateValue) { bitrate = bitrateValue; }

void LiveOutputRTMP::setBufferSize(int newBufferSize) { bufferSize = newBufferSize; }

void LiveOutputRTMP::setMinBitrate(int newMinBitrate) { minBitrate = newMinBitrate; }

void LiveOutputRTMP::setTargetUsage(int tu) { targetUsage = tu; }

void LiveOutputRTMP::setPubUser(const QString& user) { pubUser = user; }
void LiveOutputRTMP::setPubPasswd(const QString& passwd) { pubPassword = passwd; }

void LiveOutputRTMP::setCbrPaddingEnabled(bool enabled) { cbrPaddingEnabled = enabled; }

QWidget* LiveOutputRTMP::createStatusWidget(QWidget* const parent) {
  statusWidget = new StatusWidget(parent);
  return statusWidget;
}

bool LiveOutputRTMP::isAnOutputForAdvancedUser() const { return true; }

int LiveOutputRTMP::getDownsamplingFactor() const { return downsampling; }

void LiveOutputRTMP::setDownsamplingFactor(const unsigned int dsValue) { downsampling = dsValue; }

QPixmap LiveOutputRTMP::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/stream-rtp.png"); }

void LiveOutputRTMP::fillOutputValues(const VideoStitch::Ptv::Value* config,
                                      const VideoStitch::Core::PanoDefinition* panoDefinition) {
  std::string fileName;
  bool success = VideoStitch::Parse::populateString("Ptv", *config, "filename", fileName, true) ==
                 VideoStitch::Parse::PopulateResult_Ok;
  setUrl(success ? QString::fromStdString(fileName) : DEFAULT_RTMP);

  // create cod_enc from encoder & codec for legacy reason
  std::stringstream cod_enc;
  std::string cod;
  if (VideoStitch::Parse::populateString("Ptv", *config, "codec", cod, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    cod_enc << cod;
  } else {
    cod_enc << DEFAULT_CODEC;
  }

  std::string enc;
  if (VideoStitch::Parse::populateString("Ptv", *config, "encoder", enc, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    cod_enc << "_" << enc;
  }

  std::string video_codec = cod_enc.str();
  VideoStitch::Parse::populateString("Ptv", *config, "video_codec", video_codec, true);

  createEncoderList(QString::fromStdString(video_codec));
  std::string ps;
  if (VideoStitch::Parse::populateString("Ptv", *config, "preset", ps, true) == VideoStitch::Parse::PopulateResult_Ok) {
    preset = QString::fromStdString(ps);
  } else {
    preset = DEFAULT_PRESET;
  }
  std::string stdTune;
  if (VideoStitch::Parse::populateString("Ptv", *config, "tune", stdTune, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    tune = QString::fromStdString(stdTune);
  }

  if (video_codec.find("nvenc") != std::string::npos) {
    if (preset == "ll") {
      tune = H264Config::TuneEnum::getDescriptorFromEnum(H264Config::ZEROLATENCY);
      preset = H264Config::PresetEnum::getDescriptorFromEnum(H264Config::MEDIUM);
    } else if (preset == "llhp") {
      tune = H264Config::TuneEnum::getDescriptorFromEnum(H264Config::ZEROLATENCY);
      preset = H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FAST);
    } else if (preset == "llhq") {
      tune = H264Config::TuneEnum::getDescriptorFromEnum(H264Config::ZEROLATENCY);
      preset = H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOW);
    }
  }

  std::string stdProfile;
  if (VideoStitch::Parse::populateString("Ptv", *config, "profile", stdProfile, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    profile = QString::fromStdString(stdProfile);
  }

  if (VideoStitch::Parse::populateInt("Ptv", *config, "downsampling_factor", downsampling, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    downsampling = DEFAULT_DOWNSAMPLING;
  }

  int resizedWidth = DEFAULT_PANO_WIDTH;
  int resizedHeight = DEFAULT_PANO_HEIGHT;
  if (panoDefinition) {
    resizedWidth = int(panoDefinition->getWidth()) / downsampling;
    resizedHeight = int(panoDefinition->getHeight()) / downsampling;
  }

  auto pixelRate = VideoStitch::getPixelRate(resizedWidth, resizedHeight);

  std::string stdLevel;
  if (VideoStitch::Parse::populateString("Ptv", *config, "level", stdLevel, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    level = QString::fromStdString(stdLevel);
  } else {
    level = VideoStitch::getLevelFromMacroblocksRate(VideoStitch::getMacroblocksRate(pixelRate),
                                                     VideoStitch::VideoCodec::getValueFromDescriptor(encoder));
  }

  if (VideoStitch::Parse::populateInt("Ptv", *config, "bitrate", bitrate, true) !=
      VideoStitch::Parse::PopulateResult::OK) {
    bitrate = VideoStitch::getDefaultBitrate(pixelRate);
  }

  std::string user;
  VideoStitch::Parse::populateString("Ptv", *config, "pub_user", user, false);
  pubUser = QString::fromStdString(user);

  std::string brMode;
  if (VideoStitch::Parse::populateString("Ptv", *config, "bitrate_mode", brMode, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    if (BitRateModeEnum::getEnumFromDescriptor(QString::fromStdString(brMode).toUpper()).getValue() ==
        BitRateMode::CUSTOM) {
      BitRateModeEnum::setDescriptorForEnum(BitRateMode::CUSTOM, QString::fromStdString(brMode).toUpper());
    }
    bitrateMode = BitRateModeEnum::getEnumFromDescriptor(QString::fromStdString(brMode).toUpper());
  }

  if (VideoStitch::Parse::populateInt("Ptv", *config, "quality_balance", qualityBalance, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    qualityBalance = DEFAULT_QUALITY_BALANCE;
  }
  VideoStitch::Parse::populateInt("Ptv", *config, "qp", qp, false);

  if (VideoStitch::Parse::populateInt("Ptv", *config, "b_frames", bframes, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    bframes = DEFAULT_B_FRAMES;
  }

  VideoStitch::Parse::populateInt("Ptv", *config, "buffer_size", bufferSize, false);
  VideoStitch::Parse::populateBool("Ptv", *config, "cbr_padding", cbrPaddingEnabled, false);
  VideoStitch::Parse::populateInt("Ptv", *config, "bitrate_min", minBitrate, false);
  VideoStitch::Parse::populateInt("Ptv", *config, "vbvMaxBitrate", maxBitrate, false);

  if (VideoStitch::Parse::populateInt("Ptv", *config, "target_usage", targetUsage, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    targetUsage = DEFAULT_TARGET_USAGE;
  }

  std::string stdMemType;
  if (VideoStitch::Parse::populateString("Ptv", *config, "mem_type", stdMemType, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    memType = QString::fromStdString(stdMemType);
  }

  VideoStitch::Parse::populateInt("Ptv", *config, "gop", gop, true);
}

std::vector<std::string> LiveOutputRTMP::getSupportedVideoCodecs() {
  QString cardName = VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::RTMP);
  return PluginsController::listVideoCodecs(cardName);
}

/* Create List of available (encoder, codec) pair
   taking into account a potential custom one set in the config file */
void LiveOutputRTMP::createEncoderList(QString codec) {
  VideoStitch::VideoCodec::VideoCodecEnum encoderEnum = VideoStitch::VideoCodec::getEnumFromString(codec);

  for (auto scodec : getSupportedVideoCodecs()) {
    encoderList.append(VideoStitch::VideoCodec::VideoEncoderEnum::getDescriptorFromEnum(
        VideoStitch::VideoCodec::getEnumFromString(QString::fromStdString(scodec))));
  }
  /* HEVC is not officially supported by RTMP, do not propose it except if specified by the user */
  if (encoderEnum != VideoStitch::VideoCodec::VideoCodecEnum::NVENC_HEVC) {
    encoderList.removeAll(VideoStitch::VideoCodec::VideoEncoderEnum::getDescriptorFromEnum(
        VideoStitch::VideoCodec::VideoCodecEnum::NVENC_HEVC));
  }
  /* If a custom encoder was set in the config file, update the CUSTOM descriptor to display & keep these settings */
  if (encoderEnum == VideoStitch::VideoCodec::VideoCodecEnum::UNKNOWN) {
    VideoStitch::VideoCodec::VideoEncoderEnum::setDescriptorForEnum(
        VideoStitch::VideoCodec::VideoCodecEnum::UNKNOWN,
        QStringList({QString("%0 : %1")
                         .arg(VideoStitch::VideoCodec::VideoEncoderEnum::getDescriptorFromEnum(
                                  VideoStitch::VideoCodec::VideoCodecEnum::UNKNOWN)
                                  .at(0))
                         .arg(codec),
                     codec}));
    encoderList.append(VideoStitch::VideoCodec::VideoEncoderEnum::getDescriptorFromEnum(
        VideoStitch::VideoCodec::VideoCodecEnum::UNKNOWN));
  }

  encoder = VideoStitch::VideoCodec::VideoEncoderEnum::getDescriptorFromEnum(encoderEnum);
}

VideoStitch::Ptv::Value* LiveOutputRTMP::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = VideoStitch::OutputFormat::getStringFromEnum(type).toStdString();
  value->get("video_codec")->asString() = encoder.at(1).toStdString();
  if ((encoder.at(1).toStdString().find("nvenc") != std::string::npos) &&
      (tune == (H264Config::TuneEnum::getDescriptorFromEnum(H264Config::ZEROLATENCY)))) {
    if (preset == H264Config::PresetEnum::getDescriptorFromEnum(H264Config::MEDIUM)) {
      value->get("preset")->asString() = "ll";
    } else if (preset == H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FAST)) {
      value->get("preset")->asString() = "llhp";
    } else if (preset == H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOW)) {
      value->get("preset")->asString() = "llhq";
    } else {
      value->get("preset")->asString() = preset.toStdString();
    }
  } else {
    value->get("preset")->asString() = preset.toStdString();
  }
  value->get("tune")->asString() = tune.toStdString();
  value->get("profile")->asString() = profile.toStdString();
  value->get("level")->asString() = level.toStdString();
  value->get("filename")->asString() = url.toStdString();
  value->get("bitrate_mode")->asString() = bitrateMode.getDescriptor().toStdString();
  value->get("quality_balance")->asInt() = qualityBalance;
  if (qp > -1) {
    value->get("qp")->asInt() = qp;
  }
  value->get("b_frames")->asInt() = bframes;
  value->get("bitrate")->asInt() = bitrate;
  if (bufferSize != -1) {
    value->get("buffer_size")->asInt() = bufferSize;
  }
  value->get("cbr_padding")->asBool() = cbrPaddingEnabled;
  if (minBitrate != -1) {
    value->get("bitrate_min")->asInt() = minBitrate;
  }
  if (maxBitrate > -1) {
    value->get("vbvMaxBitrate")->asInt() = qMax(maxBitrate, bitrate);
  }
  value->get("target_usage")->asInt() = targetUsage;
  if (!memType.isEmpty()) {
    value->get("mem_type")->asString() = memType.toStdString();
  }

  value->get("gop")->asInt() = gop;
  if (needsAuthentication()) {
    value->get("pub_user")->asString() = pubUser.toStdString();
  }

  if (hasLog) {
    const QString logFile = initializeAvStatsLogFile("rtmp");
    value->get("stats_log_file")->asString() = logFile.toStdString();
  }

  if (downsampling > 1) {
    value->get("downsampling_factor")->asInt() = downsampling;
  }
  audioConfig->serializeIn(value);
  return value;
}

VideoStitch::Ptv::Value* LiveOutputRTMP::serializePrivate() const {
  VideoStitch::Ptv::Value* value = serialize();
  value->get("user_agent")->asString() = QCoreApplication::applicationName().toStdString();
  value->get("pub_passwd")->asString() = pubPassword.toStdString();
  return value;
}

QString LiveOutputRTMP::getUrl() const { return url; }

QString LiveOutputRTMP::initializeAvStatsLogFile(QString name) {
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

void LiveOutputRTMP::connectNotifications(VideoStitch::Output::Output& outputWriter) {
  // connect the state notifier to the status widget

  outputWriter.getOutputEventManager().subscribe(VideoStitch::Output::OutputEventManager::EventType::Connected,
                                                 [this](const std::string& status) {
                                                   emit statusWidget->connected(status);

                                                   if (this->state == OutputState::DISABLED) {
                                                     return;
                                                   }
                                                   this->state = OutputState::ENABLED;
                                                 });

  auto noConnection = [this]() {
    if (this->state == OutputState::DISABLED) {
      return;
    }
    this->state = OutputState::CONNECTING;
  };

  outputWriter.getOutputEventManager().subscribe(VideoStitch::Output::OutputEventManager::EventType::Connecting,
                                                 [this, noConnection](const std::string&) {
                                                   emit statusWidget->connecting();

                                                   noConnection();
                                                 });

  outputWriter.getOutputEventManager().subscribe(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                                 [this, noConnection](const std::string&) {
                                                   emit statusWidget->disconnected();

                                                   noConnection();
                                                 });
}

VideoStitch::Potential<VideoStitch::Output::Output> LiveOutputRTMP::createWriter(LiveProjectDefinition* project,
                                                                                 VideoStitch::FrameRate framerate) {
  // create the callback
  VideoStitch::Potential<VideoStitch::Output::Output> writer = VideoStitch::Output::create(
      *serializePrivate(), getIdentifier().toStdString(), project->getPanoConst()->getWidth(),
      project->getPanoConst()->getHeight(), framerate,
      VideoStitch::Audio::getSamplingRateFromInt(audioConfig->getSamplingRate()),
      VideoStitch::Audio::getSamplingDepthFromString(audioConfig->getSamplingFormat().toStdString().c_str()),
      VideoStitch::Audio::getChannelLayoutFromString(audioConfig->getChannelLayout().toStdString().c_str()));
  if (writer.ok()) {
    connectNotifications(*writer.object());
    writer->init();
  } else {
    statusWidget->disconnected();
  }
  return writer;
}

OutputConfigurationWidget* LiveOutputRTMP::createConfigurationWidget(QWidget* const parent) {
  return new ConfigurationOutputStreaming(this, parent);
}

StatusWidget::StatusWidget(QWidget* const parent)
    : QWidget(parent),
      layout(new QHBoxLayout(this)),
      labelStreaming(new QLabel(tr("Connecting RTMP"), this)),
      labelAnimation(new QLabel(this)),
      movieAnimation(new QMovie(LOADING_ICON, nullptr, this)) {
  labelAnimation->setFixedSize(STATUS_ICON_SIZE, STATUS_ICON_SIZE);
  labelAnimation->setMovie(movieAnimation);
  movieAnimation->start();
  labelStreaming->setFixedHeight(STATUS_ICON_SIZE);
  layout->addWidget(labelAnimation);
  layout->addWidget(labelStreaming);
}

StatusWidget::~StatusWidget() {}

void StatusWidget::connected(const std::string& status) {
  labelStreaming->setText(tr("LIVE (%0)").arg(QString::fromStdString(status)));
}

void StatusWidget::connecting() { labelStreaming->setText(tr("Connecting RTMP")); }

void StatusWidget::disconnected() { labelStreaming->setText(tr("Disconnected")); }
