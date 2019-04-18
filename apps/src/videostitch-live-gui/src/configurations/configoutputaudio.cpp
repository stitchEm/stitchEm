// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configoutputaudio.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "videostitcher/liveaudio.hpp"
#include "guiconstants.hpp"

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

ConfigOutputAudio::ConfigOutputAudio(QWidget* const parent)
    : QWidget(parent), liveOutputFactory(nullptr), audioConfig(nullptr) {
  setupUi(this);
  connect(comboAudioCodec, SIGNAL(currentIndexChanged(int)), this, SLOT(onCodecConfigChanged()));

  connect(comboAudioCodec, SIGNAL(currentIndexChanged(int)), this, SIGNAL(notifyConfigChanged()));
  connect(audioSamplingRate, SIGNAL(currentIndexChanged(int)), this, SIGNAL(notifyConfigChanged()));
  connect(audioChannelLayout, SIGNAL(currentIndexChanged(int)), this, SIGNAL(notifyConfigChanged()));
  connect(comboBitrate, SIGNAL(currentIndexChanged(int)), this, SIGNAL(notifyConfigChanged()));
  loadStaticValues();
}

ConfigOutputAudio::~ConfigOutputAudio() {}

void ConfigOutputAudio::setLiveAudio(const LiveOutputFactory* newLiveOutputFactory) {
  liveOutputFactory = newLiveOutputFactory;
  audioConfig = liveOutputFactory->getAudioConfig();
}

void ConfigOutputAudio::setType(const VideoStitch::OutputFormat::OutputFormatEnum type) { outputType = type; }

void ConfigOutputAudio::loadParameters() {
  const StitcherController* controller = GlobalController::getInstance().getController();
  bool hasInputAudio = controller->hasInputAudio();
  audioBox->setVisible(hasInputAudio);
  labelAudioTitle->setVisible(!hasInputAudio);

  Q_ASSERT(audioConfig != nullptr);

  onCodecConfigChanged();  // To fill the channel layout and sampling rate boxes
  if (hasInputAudio) {
    comboAudioCodec->setCurrentText(audioConfig->getAudioCodec());
    audioSamplingRate->setCurrentText(VideoStitch::AudioHelpers::getSampleRateString(
        VideoStitch::Audio::getSamplingRateFromInt(audioConfig->getSamplingRate())));
    audioChannelLayout->setCurrentText(audioConfig->getChannelLayout());
    comboBitrate->setCurrentText(QString::number(audioConfig->getBitrate()));
    switch (outputType) {
      case VideoStitch::OutputFormat::OutputFormatEnum::AJA:
      case VideoStitch::OutputFormat::OutputFormatEnum::DECKLINK:
        audioSamplingRate->setCurrentText(
            VideoStitch::AudioHelpers::getSampleRateString(VideoStitch::Audio::SamplingRate::SR_48000));
        audioChannelLayout->setCurrentText(
            VideoStitch::Audio::getStringFromChannelLayout(VideoStitch::Audio::ChannelLayout::STEREO));
        comboAudioCodec->setVisible(false);
        labelAudioCodec->setVisible(false);
        comboBitrate->setVisible(false);
        labelBitrate->setVisible(false);
        audioSamplingRate->setDisabled(true);
        audioChannelLayout->setDisabled(true);
        break;
      case VideoStitch::OutputFormat::OutputFormatEnum::MP4:
      case VideoStitch::OutputFormat::OutputFormatEnum::MOV:
      case VideoStitch::OutputFormat::OutputFormatEnum::JPG:
      case VideoStitch::OutputFormat::OutputFormatEnum::PNG:
      case VideoStitch::OutputFormat::OutputFormatEnum::PPM:
      case VideoStitch::OutputFormat::OutputFormatEnum::PAM:
      case VideoStitch::OutputFormat::OutputFormatEnum::RAW:
      case VideoStitch::OutputFormat::OutputFormatEnum::TIF:
      case VideoStitch::OutputFormat::OutputFormatEnum::YUV420P:
      case VideoStitch::OutputFormat::OutputFormatEnum::RTMP:
      case VideoStitch::OutputFormat::OutputFormatEnum::YOUTUBE:
      case VideoStitch::OutputFormat::OutputFormatEnum::OCULUS:
      case VideoStitch::OutputFormat::OutputFormatEnum::STEAMVR:
      case VideoStitch::OutputFormat::OutputFormatEnum::CUSTOM:
      case VideoStitch::OutputFormat::OutputFormatEnum::UNKNOWN:
        // Nothing specific to do for those type of output
        break;
    }
  }
}

void ConfigOutputAudio::displayMessage(QString message) {
  widget->setVisible(false);
  labelAudioTitle->setVisible(true);
  labelAudioTitle->setText(message);
}

void ConfigOutputAudio::loadStaticValues() {
  // These are the values supported by our RTMP output plugin, we should have an audio discovery for that
  // Because it could be different for other output plugins
  comboAudioCodec->addItem(
      VideoStitch::AudioHelpers::getStringFromCodec(VideoStitch::AudioHelpers::AudioCodecEnum::MP3));
  comboAudioCodec->addItem(
      VideoStitch::AudioHelpers::getStringFromCodec(VideoStitch::AudioHelpers::AudioCodecEnum::AAC));
  comboBitrate->addItems(VideoStitch::AudioHelpers::getAudioBitrates());
}

void ConfigOutputAudio::saveConfiguration() const {
  if (widget->isVisible()) {
    const VideoStitch::AudioHelpers::AudioCodecEnum audioCodecType =
        VideoStitch::AudioHelpers::getCodecFromString(comboAudioCodec->currentText());
    VideoStitch::Audio::SamplingDepth samplingDepth = liveOutputFactory->getPreferredSamplingDepth(audioCodecType);

    audioConfig->setAudioCodec(comboAudioCodec->currentText());
    audioConfig->setSamplingFormat(VideoStitch::Audio::getStringFromSamplingDepth(samplingDepth));
    audioConfig->setSamplingRate(VideoStitch::Audio::getIntFromSamplingRate(
        VideoStitch::Audio::SamplingRate(audioSamplingRate->currentData().toInt())));
    audioConfig->setChannelLayout(audioChannelLayout->currentText());
    audioConfig->setBitrate(comboBitrate->currentText().toInt());
  } else if (labelAudioTitle->isVisible()) {
    audioConfig->setAudioCodec(QString());
    audioConfig->setSamplingFormat(QString());
    audioConfig->setSamplingRate(0);
    audioConfig->setChannelLayout(QString());
    audioConfig->setBitrate(0);
  }
}

void ConfigOutputAudio::onCodecConfigChanged() {
  if (!liveOutputFactory || !audioConfig) {
    return;
  }

  QString oldChannelLayout = audioChannelLayout->currentText();
  QString oldSamplingRate = audioSamplingRate->currentText();
  audioChannelLayout->clear();
  audioSamplingRate->clear();

  const StitcherController* controller = GlobalController::getInstance().getController();
  if (!controller->hasInputAudio()) {
    return;
  }

  const VideoStitch::AudioHelpers::AudioCodecEnum audioCodecType =
      VideoStitch::AudioHelpers::getCodecFromString(comboAudioCodec->currentText());
  QList<VideoStitch::Audio::ChannelLayout> channelLayouts =
      liveOutputFactory->getSupportedChannelLayouts(audioCodecType);
  QList<VideoStitch::Audio::SamplingRate> samplingRates = liveOutputFactory->getSupportedSamplingRates(audioCodecType);

  for (VideoStitch::Audio::ChannelLayout channelLayout : channelLayouts) {
    audioChannelLayout->addItem(VideoStitch::Audio::getStringFromChannelLayout(channelLayout));
  }
  for (VideoStitch::Audio::SamplingRate samplingRate : samplingRates) {
    audioSamplingRate->addItem(VideoStitch::AudioHelpers::getSampleRateString(samplingRate), int(samplingRate));
  }

  audioChannelLayout->setCurrentText(oldChannelLayout);
  audioSamplingRate->setCurrentText(oldSamplingRate);
}
