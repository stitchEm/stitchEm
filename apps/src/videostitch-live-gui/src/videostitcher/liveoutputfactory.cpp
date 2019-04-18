// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputfactory.hpp"

#include "liveaudio.hpp"
#include "liveprojectdefinition.hpp"
#include "liveoutputfile.hpp"
#include "liveoutputrtmp.hpp"
#ifdef ENABLE_YOUTUBE_OUTPUT
#include "liveoutputyoutube.hpp"
#endif
#include "liveoutputdecklink.hpp"
#include "liveoutputoculus.hpp"
#include "liveoutputcustom.hpp"
#include "liveoutputaja.hpp"
#if defined(Q_OS_WIN)
#include "liveoutputsteamvr.hpp"
#endif

#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "guiconstants.hpp"

#include <QDir>
#include <QStandardPaths>
#include <QLabel>

LiveOutputFactory* LiveOutputFactory::createOutput(VideoStitch::Ptv::Value* config,
                                                   const VideoStitch::Core::PanoDefinition* panoDefinition) {
  std::unique_ptr<VideoStitch::Ptv::Value> ownedConfig(config);
  std::string typeString;
  LiveOutputFactory* output = nullptr;
  if (VideoStitch::Parse::populateString("Ptv", *ownedConfig, "type", typeString, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    const VideoStitch::OutputFormat::OutputFormatEnum type =
        VideoStitch::OutputFormat::getEnumFromString(QString::fromStdString(typeString));
    return createOutput(ownedConfig.release(), type, panoDefinition);
  }
  return output;
}

LiveOutputFactory* LiveOutputFactory::createOutput(VideoStitch::Ptv::Value* config,
                                                   const VideoStitch::OutputFormat::OutputFormatEnum& type,
                                                   const VideoStitch::Core::PanoDefinition* panoDefinition) {
  std::unique_ptr<const VideoStitch::Ptv::Value> ownedConfig(config);
  LiveOutputFactory* output = nullptr;
  if (VideoStitch::OutputFormat::isFileFormat(type)) {
    output = new LiveOutputFile(ownedConfig.get(), panoDefinition, type);
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::RTMP) {
    output = new LiveOutputRTMP(ownedConfig.get(), panoDefinition, type);
#ifdef ENABLE_YOUTUBE_OUTPUT
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::YOUTUBE) {
    output = new LiveOutputYoutube(ownedConfig.get(), panoDefinition, type);
#endif
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::DECKLINK) {
    output = new LiveOutputDecklink(ownedConfig.get(), type);
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::OCULUS) {
    output = new LiveRendererOculus(type);
#if defined(Q_OS_WIN)
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::STEAMVR) {
    output = new LiveOutputSteamVR(type);
#endif
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::AJA) {
    output = new LiveOutputAJA(ownedConfig.get(), type);
  } else if (type == VideoStitch::OutputFormat::OutputFormatEnum::CUSTOM) {
    output = new LiveOutputCustom(ownedConfig.get(), type);
  }

  if (output) {
    // We can't do this in the constructor since we use a virtual function of LiveOutputFactory
    output->initializeAudioOutput(ownedConfig.get());
  }
  return output;
}

LiveOutputFactory::LiveOutputFactory(VideoStitch::OutputFormat::OutputFormatEnum type)
    : audioConfig(new LiveAudio()), hasLog(false), state(OutputState::DISABLED), type(type) {}

LiveOutputFactory::~LiveOutputFactory() {}

LiveWriterFactory::LiveWriterFactory(VideoStitch::OutputFormat::OutputFormatEnum type) : LiveOutputFactory(type) {}

VideoStitch::Potential<VideoStitch::Output::Output> LiveWriterFactory::createWriter(LiveProjectDefinition* project,
                                                                                    VideoStitch::FrameRate framerate) {
  return VideoStitch::Output::create(*serialize(), getIdentifier().toStdString(), project->getPanoConst()->getWidth(),
                                     project->getPanoConst()->getHeight(), framerate);
}

VideoStitch::Potential<VideoStitch::Output::StereoWriter> LiveWriterFactory::createStereoWriter(
    LiveProjectDefinition* project, VideoStitch::FrameRate framerate) {
  VideoStitch::Potential<VideoStitch::Output::Output> w = createWriter(project, framerate);
  return VideoStitch::Output::StereoWriter::createComposition(
      w.release()->getVideoWriter(), VideoStitch::Output::StereoWriter::VerticalLayout, Device);
}

OutputConfigurationWidget* LiveOutputFactory::createConfigurationWidget(QWidget* const parent) {
  Q_UNUSED(parent);
  return nullptr;
}

bool LiveOutputFactory::checkIfIsActivable(const VideoStitch::Core::PanoDefinition* panoDefinition,
                                           QString& message) const {
  Q_UNUSED(panoDefinition);
  Q_UNUSED(message);
  return true;
}

const QString LiveOutputFactory::getOutputTypeDisplayName() const {
  return VideoStitch::OutputFormat::getDisplayNameFromEnum(type);
}

QList<VideoStitch::Audio::SamplingDepth> LiveOutputFactory::getOrderedSamplingDepths() {
  QList<VideoStitch::Audio::SamplingDepth> orderedSamplingDepths;
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::DBL_P);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::DBL);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::FLT);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::FLT_P);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::INT32);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::INT32_P);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::INT16);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::INT16_P);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::UINT8);
  orderedSamplingDepths.append(VideoStitch::Audio::SamplingDepth::UINT8_P);

  return orderedSamplingDepths;
}

QList<VideoStitch::Audio::SamplingDepth> LiveOutputFactory::getSupportedSamplingDepths(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const {
  Q_UNUSED(audioCodec);
  return QList<VideoStitch::Audio::SamplingDepth>();
}

VideoStitch::Audio::SamplingDepth LiveOutputFactory::getPreferredSamplingDepth(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodecType) const {
  QList<VideoStitch::Audio::SamplingDepth> orderedSamplingDepths = getOrderedSamplingDepths();
  QList<VideoStitch::Audio::SamplingDepth> supportedSamplingDepths = getSupportedSamplingDepths(audioCodecType);
  for (VideoStitch::Audio::SamplingDepth samplingDepth : orderedSamplingDepths) {
    if (supportedSamplingDepths.contains(samplingDepth)) {
      return samplingDepth;
    }
  }
  return VideoStitch::Audio::SamplingDepth::SD_NONE;
}

QList<VideoStitch::Audio::SamplingRate> LiveOutputFactory::getSupportedSamplingRates(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const {
  QList<VideoStitch::Audio::SamplingRate> samplingRates;
  switch (audioCodec) {
    case VideoStitch::AudioHelpers::AudioCodecEnum::MP3:
      samplingRates << VideoStitch::Audio::SamplingRate::SR_44100;
      break;
    case VideoStitch::AudioHelpers::AudioCodecEnum::AAC:
    default:
      samplingRates << VideoStitch::Audio::SamplingRate::SR_48000;
      break;
  }
  return samplingRates;
}

QList<VideoStitch::Audio::ChannelLayout> LiveOutputFactory::getSupportedChannelLayouts(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodecType) const {
  VideoStitch::Audio::SamplingDepth samplingDepth = getPreferredSamplingDepth(audioCodecType);
  QList<VideoStitch::Audio::ChannelLayout> channelLayouts;
  if ((audioCodecType == VideoStitch::AudioHelpers::AudioCodecEnum::MP3 &&
       samplingDepth == VideoStitch::Audio::SamplingDepth::INT16) ||
      audioCodecType == VideoStitch::AudioHelpers::AudioCodecEnum::AAC) {
    channelLayouts << VideoStitch::Audio::STEREO;
    // IBC-demo
    channelLayouts << VideoStitch::Audio::AMBISONICS_WXYZ;
  } else {
    channelLayouts << VideoStitch::Audio::MONO << VideoStitch::Audio::STEREO;
  }
  return channelLayouts;
}

LiveOutputFactory::PanoSizeChange LiveOutputFactory::supportPanoSizeChange(int newWidth, int newHeight) const {
  Q_UNUSED(newWidth);
  Q_UNUSED(newHeight);
  return PanoSizeChange::Supported;
}

QString LiveOutputFactory::getPanoSizeChangeDescription(int newWidth, int newHeight) const {
  Q_UNUSED(newWidth);
  Q_UNUSED(newHeight);
  return QString();
}

void LiveOutputFactory::updateForPanoSizeChange(int newWidth, int newHeight) {
  Q_UNUSED(newWidth);
  Q_UNUSED(newHeight);
}

QLabel* LiveOutputFactory::createStatusIcon(QWidget* const parent) const {
  QLabel* label = new QLabel(parent);
  label->setScaledContents(true);
  label->setFixedSize(STATUS_ICON_SIZE, STATUS_ICON_SIZE);
  label->setPixmap(getIcon());
  label->setToolTip(getIdentifier());
  return label;
}

void LiveOutputFactory::initializeAudioOutput(const VideoStitch::Ptv::Value* config) const {
  audioConfig->initialize(config, *this);
}

LiveRendererFactory::LiveRendererFactory(VideoStitch::OutputFormat::OutputFormatEnum type) : LiveOutputFactory(type) {}
