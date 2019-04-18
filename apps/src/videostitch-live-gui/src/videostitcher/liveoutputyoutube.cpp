// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputyoutube.hpp"

#include "googleauthenticationmanager.hpp"
#include "youtubebroadcastmodel.hpp"
#include "liveaudio.hpp"

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include "libvideostitch-gui/utils/samplingrateenum.hpp"

LiveOutputYoutube::LiveOutputYoutube(const VideoStitch::Ptv::Value* config,
                                     const VideoStitch::Core::PanoDefinition* panoDefinition,
                                     const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveOutputRTMP(config, panoDefinition, type) {
  type = QStringLiteral("youtube");
  setBitRateMode(BitRateModeEnum(BitRateMode::CBR));
  fillOutputValues(config, panoDefinition);
}

VideoStitch::Ptv::Value* LiveOutputYoutube::serialize() const {
  auto value = LiveOutputRTMP::serialize();

  value->get("broadcastId")->asString() = broadcastId.toStdString();

  return value;
}

bool LiveOutputYoutube::showParameters() const { return false; }

bool LiveOutputYoutube::needsAuthentication() const { return true; }

bool LiveOutputYoutube::forceConstantBitRate() const { return true; }

QPixmap LiveOutputYoutube::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/youtube.png"); }

bool LiveOutputYoutube::checkIfIsActivable(const VideoStitch::Core::PanoDefinition* panoDefinition,
                                           QString& message) const {
  Q_UNUSED(panoDefinition);
  bool authorized = GoogleAuthenticationManager::getInstance().authorizeClient(getPubUser());
  if (!authorized) {
    message = tr("Please authorize %0 to check the stream.").arg(QCoreApplication::applicationName());
    return false;
  }

  if (!YoutubeBroadcastModel::isBroadcastStreamable(GoogleAuthenticationManager::getInstance().getCredential(),
                                                    broadcastId, message)) {
    return false;
  }

  return true;
}

bool LiveOutputYoutube::isAnOutputForAdvancedUser() const { return false; }

QString LiveOutputYoutube::getBroadcastId() const { return broadcastId; }

void LiveOutputYoutube::setBroadcastId(const QString& value) { broadcastId = value; }

void LiveOutputYoutube::initializeAudioOutput(const VideoStitch::Ptv::Value* config) const {
  LiveOutputFactory::initializeAudioOutput(config);
  audioConfig->setSamplingRate(VideoStitch::Audio::getIntFromSamplingRate(VideoStitch::Audio::SamplingRate::SR_44100));
  audioConfig->setAudioCodec(QString::fromLatin1("mp3"));
}

void LiveOutputYoutube::fillOutputValues(const VideoStitch::Ptv::Value* config,
                                         const VideoStitch::Core::PanoDefinition* /*panoDefinition*/) {
  std::string stdBroadcastId;
  if (VideoStitch::Parse::populateString("Ptv", *config, "broadcastId", stdBroadcastId, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    broadcastId = QString::fromStdString(stdBroadcastId);
  }
}
