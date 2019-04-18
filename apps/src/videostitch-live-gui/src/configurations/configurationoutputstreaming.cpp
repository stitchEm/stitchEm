// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configurationoutputstreaming.hpp"
#include "ui_configurationoutputstreaming.h"

#include "videostitcher/encodedoutputhelpers.hpp"
#include "videostitcher/liveoutputrtmp.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "youtubeconfiguration.hpp"
#include "istreamingserviceconfiguration.hpp"

#include "libvideostitch-gui/utils/h264settingsenum.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

ConfigurationOutputStreaming::ConfigurationOutputStreaming(LiveOutputRTMP* outputref, QWidget* const parent)
    : OutputConfigurationWidget(parent),
      outputRef(outputref),
      streamingServiceConfig(
          IStreamingServiceConfiguration::createStreamingService(this, outputref, projectDefinition)) {
  setupUi(this);

  connect(comboReductions, &QComboBox::currentTextChanged, this, &ConfigurationOutputStreaming::updateEncodingSettings);
  connect(profileBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputStreaming::updateBitrateMaximum);
  connect(comboBitrateModes, &QComboBox::currentTextChanged, this,
          &ConfigurationOutputStreaming::updateBitrateWidgetsVisibility);
  connect(encoderBox, &QComboBox::currentTextChanged, this,
          &ConfigurationOutputStreaming::updatePresetWidgetsVisibility);
  connect(useCustomBufferSizeBox, &QCheckBox::toggled, this, &ConfigurationOutputStreaming::setBufferSizeToDefault);
  connect(useAutomaticBitrateBox, &QCheckBox::toggled, this, &ConfigurationOutputStreaming::setMinBitrateToDefault);

  connect(comboReductions, &QComboBox::currentTextChanged, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(streamingServiceConfig, &IStreamingServiceConfiguration::stateChanged, this,
          &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(streamingServiceConfig, &IStreamingServiceConfiguration::basicConfigurationComplete, this,
          &ConfigurationOutputStreaming::onBasicConfigurationComplete);
  connect(streamingServiceConfig, &IStreamingServiceConfiguration::basicConfigurationCanceled, this,
          &ConfigurationOutputStreaming::onBasicConfigurationCanceled);
  connect(encoderBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(presetBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(profileBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(comboBitrateModes, &QComboBox::currentTextChanged, this,
          &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(qualityBalanceSlider, &QSlider::valueChanged, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(spinBitRate, SIGNAL(valueChanged(int)), this, SLOT(updateMinBitrateMaximum()));
  connect(spinBitRate, SIGNAL(valueChanged(int)), this, SLOT(onConfigurationChanged()));
  connect(useCustomBufferSizeBox, &QCheckBox::toggled, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(bufferSizeBox, SIGNAL(valueChanged(int)), this, SLOT(onConfigurationChanged()));
  connect(useAutomaticBitrateBox, &QCheckBox::toggled, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(minBitrateBox, SIGNAL(valueChanged(int)), this, SLOT(onConfigurationChanged()));
  connect(cbrPaddingBox, &QCheckBox::toggled, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(lowLatencyBox, &QCheckBox::toggled, this, &ConfigurationOutputStreaming::onConfigurationChanged);
  connect(spinGOP, SIGNAL(valueChanged(int)), this, SLOT(onConfigurationChanged()));
  connect(audioConfig, &ConfigOutputAudio::notifyConfigChanged, this,
          &ConfigurationOutputStreaming::onConfigurationChanged);

  serviceSettingsLayout->addWidget(streamingServiceConfig);
  streamingServiceConfig->show();

  verticalLayout->addLayout(buttonsLayout);

  groupBoxGeneralSettings->setVisible(outputRef->showParameters());
  groupBoxEncoderSettings->setVisible(outputRef->showParameters());
  audioConfig->setVisible(outputRef->showParameters());
  advancedModeBox->setChecked(false);
  if (outputRef->forceConstantBitRate()) {
    comboBitrateModes->addItem(BitRateModeEnum::getDescriptorFromEnum(CBR));
  } else {
    comboBitrateModes->addItems(BitRateModeEnum::getDescriptorsList());
  }

  for (auto it : outputRef->getEncoderList()) {
    encoderBox->addItem(it.at(0), it);
  }

  profileBox->addItem(H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::BASELINE));
  profileBox->addItem(H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::MAIN));
  profileBox->addItem(H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::HIGH));
  audioConfig->setLiveAudio(outputRef);
}

ConfigurationOutputStreaming::~ConfigurationOutputStreaming() {}

void ConfigurationOutputStreaming::fillWidgetWithValue() {
  OutputConfigurationWidget::fillWidgetWithValue();

  // General settings
  comboReductions->setCurrentIndex(comboReductions->findData(outputRef->getDownsamplingFactor()));
  encoderBox->setCurrentText(outputRef->getEncoder().at(0));
  updateEncodingSettings();
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  double fps = double(frameRate.num) / double(frameRate.den);
  inputFpsLabel->setText(QString::number(fps));

  // Encoder settings
  profileBox->setCurrentText(outputRef->getProfile());
  if (!outputRef->getLevel().isEmpty()) {
    levelLabel->setText(outputRef->getLevel());
    updateBitrateMaximum();
  }
  comboBitrateModes->setCurrentText(outputRef->getBitRateMode().getDescriptor());
  qualityBalanceSlider->setValue(outputRef->getQualityBalance());
  if (outputRef->getBitRate() != -1) {
    spinBitRate->setValue(outputRef->getBitRate());
  }
  useCustomBufferSizeBox->setChecked(outputRef->getBufferSize() != -1);
  bufferSizeBox->setValue(outputRef->getBufferSize());
  useAutomaticBitrateBox->setChecked(outputRef->getMinBitrate() != -1);
  minBitrateBox->setValue(qMin(outputRef->getMinBitrate(), outputRef->getBitRate() - 1));
  cbrPaddingBox->setChecked(outputRef->cbrPaddingIsEnabled());
  lowLatencyBox->setChecked(outputRef->getTune() ==
                            H264Config::TuneEnum::getDescriptorFromEnum(H264Config::ZEROLATENCY));
  spinGOP->setValue(outputRef->getGOP());

  // Audio settings
  audioConfig->loadParameters();

  if (!streamingServiceConfig->loadConfiguration()) {
    return;
  }
}

bool ConfigurationOutputStreaming::hasValidConfiguration() const {
  return streamingServiceConfig->hasValidConfiguration();
}

void ConfigurationOutputStreaming::toggleWidgetState() {
  for (auto object : configurationWidget->findChildren<QWidget*>()) {
    object->setEnabled(!object->isEnabled());
  }
  // bitrate config can be updated while streaming
  groupBoxEncoderSettings->setEnabled(true);
  for (auto object : groupBoxEncoderSettings->findChildren<QWidget*>()) {
    object->setEnabled(!object->isEnabled());
  }
  spinBitRate->setEnabled(true);
  bitrateLabel->setEnabled(true);
  minBitrateLabel->setEnabled(true);
  minBitrateBox->setEnabled(true);
  useAutomaticBitrateBox->setEnabled(true);
  advancedModeBox->setEnabled(true);
  for (auto object : advancedModeBox->findChildren<QWidget*>()) {
    object->setEnabled(!object->isEnabled());
  }
  useCustomBufferSizeBox->setEnabled(true);
  bufferSizeBox->setEnabled(true);
  streamingServiceConfig->setEnabled(!streamingServiceConfig->isEnabled());
}

void ConfigurationOutputStreaming::updateAfterChangedMode() {
  configurationWidget->setVisible(mode == IConfigurationCategory::Mode::Edition ||
                                  outputRef->isAnOutputForAdvancedUser());

  if (mode == IConfigurationCategory::Mode::CreationInPopup || mode == IConfigurationCategory::Mode::CreationInStack) {
    streamingServiceConfig->startBaseConfiguration();
  }
}

LiveOutputFactory* ConfigurationOutputStreaming::getOutput() const { return outputRef; }

void ConfigurationOutputStreaming::saveData() {
  if (outputRef != nullptr) {
    const QString previousFileName = outputRef->getIdentifier();
    outputRef->setBitRate(spinBitRate->value());
    outputRef->setBitRateMode(BitRateModeEnum::getEnumFromDescriptor(comboBitrateModes->currentText()));
    if (qualityBalanceSlider->isVisible()) {
      outputRef->setQualityBalance(qualityBalanceSlider->value());
    }
    outputRef->setGOP(spinGOP->value());
    outputRef->setDownsamplingFactor(comboReductions->currentData().toInt());
    audioConfig->saveConfiguration();

    outputRef->setEncoder(encoderBox->currentData().toStringList());
    switch (VideoStitch::VideoCodec::getValueFromDescriptor(encoderBox->currentData().toStringList())) {
      case VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264:
        outputRef->setTargetUsage(H264Config::PresetEnum::getEnumFromDescriptor(presetBox->currentText()).getValue());
        break;
      default:
        outputRef->setPreset(presetBox->currentText());
        break;
    }
    outputRef->setProfile(profileBox->currentText());
    outputRef->setLevel(levelLabel->text());
    if (useCustomBufferSizeBox->isChecked()) {
      outputRef->setBufferSize(bufferSizeBox->value());
    } else {
      outputRef->setBufferSize(-1);
    }
    if (useAutomaticBitrateBox->isChecked()) {
      outputRef->setMinBitrate(minBitrateBox->value());
    } else {
      outputRef->setMinBitrate(-1);
    }
    outputRef->setCbrPaddingEnabled(cbrPaddingBox->isChecked());
    outputRef->setTune(lowLatencyBox->isChecked() ? H264Config::TuneEnum::getDescriptorFromEnum(H264Config::ZEROLATENCY)
                                                  : QString());

    streamingServiceConfig->saveConfiguration();

    projectDefinition->updateOutputId(previousFileName);
    emit reqChangeOutputId(previousFileName, outputRef->getIdentifier());
    if (!streamingServiceConfig->isEnabled()) {
      emit reqChangeOutputConfig(outputRef->getIdentifier());
    }
  }
}

void ConfigurationOutputStreaming::updateEncodingSettings() {
  int factor = comboReductions->currentData().toInt();
  if (factor == 0) {
    return;
  }
  int resizedWidth = int(projectDefinition->getPanoConst()->getWidth()) / factor;
  int resizedHeight = int(projectDefinition->getPanoConst()->getHeight()) / factor;

  auto pixelRate = VideoStitch::getPixelRate(resizedWidth, resizedHeight);
  levelLabel->setText(VideoStitch::getLevelFromMacroblocksRate(
      VideoStitch::getMacroblocksRate(pixelRate),
      VideoStitch::VideoCodec::getValueFromDescriptor(encoderBox->currentData().toStringList())));
  updateBitrateMaximum();
  spinBitRate->setValue(VideoStitch::getDefaultBitrate(pixelRate));
  setBufferSizeToDefault();
  setMinBitrateToDefault();
}

void ConfigurationOutputStreaming::updateBitrateMaximum() {
  QString profile = profileBox->currentText();
  QString level = levelLabel->text();
  int maxBitrate = VideoStitch::getMaxConstantBitRate(
      profile, level, VideoStitch::VideoCodec::getValueFromDescriptor(encoderBox->currentData().toStringList()));
  spinBitRate->setMaximum(maxBitrate);
}

void ConfigurationOutputStreaming::updateMinBitrateMaximum() {
  minBitrateBox->setMaximum(spinBitRate->value() - 1);
  if (useAutomaticBitrateBox->isChecked()) {
    if (minBitrateBox->value() >= spinBitRate->value()) {
      minBitrateBox->setValue(spinBitRate->value() - 1);
    }
    if ((10 * minBitrateBox->value()) < spinBitRate->value()) {
      minBitrateBox->setValue(spinBitRate->value() / 2);
    }
  }
}

void ConfigurationOutputStreaming::reactToChangedProject() {
  if (projectDefinition) {
    comboReductions->clear();
    for (unsigned int factor : projectDefinition->getDownsampledFactors()) {
      comboReductions->addItem(projectDefinition->getOutputDisplayableString(factor), factor);
    }
    streamingServiceConfig->setLiveProjectDefinition(projectDefinition);
    fillWidgetWithValue();
  }
}

void ConfigurationOutputStreaming::updatePresetWidgetsVisibility(QString) {
  presetBox->clear();
  VideoStitch::VideoCodec::VideoCodecEnum codec =
      VideoStitch::VideoCodec::getValueFromDescriptor(encoderBox->currentData().toStringList());
  switch (codec) {
    case VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264:
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::VERYSLOW));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOWER));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOW));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::MEDIUM));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FAST));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FASTER));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SUPERFAST));
      break;
    case VideoStitch::VideoCodec::VideoCodecEnum::H264:
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::VERYSLOW));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOWER));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOW));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::MEDIUM));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FAST));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FASTER));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SUPERFAST));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::ULTRAFAST));
      break;
    case VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264:
    case VideoStitch::VideoCodec::VideoCodecEnum::NVENC_HEVC:
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::SLOW));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::MEDIUM));
      presetBox->addItem(H264Config::PresetEnum::getDescriptorFromEnum(H264Config::FAST));
      break;
    case VideoStitch::VideoCodec::VideoCodecEnum::UNKNOWN:
      if (!outputRef->getPreset().isEmpty()) {
        presetBox->addItem(outputRef->getPreset());
      }
    default:
      break;
  }
  lowLatencyBox->setVisible(codec != VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264);
  useCustomBufferSizeBox->setVisible(codec != VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264);
  if (codec == VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264) {
    presetBox->setCurrentText(
        H264Config::PresetEnum::getDescriptorFromEnum((H264Config::Preset)outputRef->getTargetUsage()));
  } else if (!outputRef->getPreset().isEmpty()) {
    presetBox->setCurrentText(outputRef->getPreset());
  }
  updateEncodingSettings();
  updateBitrateWidgetsVisibility(comboBitrateModes->currentText());
}

void ConfigurationOutputStreaming::updateBitrateWidgetsVisibility(QString bitrateMode) {
  bool useCBR = BitRateModeEnum::getEnumFromDescriptor(bitrateMode) == CBR;
  bitrateLabel->setText(useCBR ? tr("Bitrate") : tr("Max bitrate"));
  switch (VideoStitch::VideoCodec::getValueFromDescriptor(encoderBox->currentData().toStringList())) {
    case VideoStitch::VideoCodec::VideoCodecEnum::H264:
      qualityBalanceLabel->setVisible(!useCBR);
      qualityBalanceSlider->setVisible(!useCBR);
      minQualityLabel->setVisible(!useCBR);
      maxQualityLabel->setVisible(!useCBR);
      cbrPaddingBox->setVisible(useCBR);
      break;
    default:
      qualityBalanceLabel->setVisible(false);
      qualityBalanceSlider->setVisible(false);
      minQualityLabel->setVisible(false);
      maxQualityLabel->setVisible(false);
      cbrPaddingBox->setVisible(false);
      break;
  }
}

void ConfigurationOutputStreaming::setBufferSizeToDefault() {
  // By default, buffer size is the same than bitrate
  bufferSizeBox->setValue(spinBitRate->value());
}

void ConfigurationOutputStreaming::setMinBitrateToDefault() {
  // By default, min bitrate is 50% of bitrate
  minBitrateBox->setValue(spinBitRate->value() / 2);
  minBitrateBox->setMaximum(spinBitRate->value() - 1);
}

void ConfigurationOutputStreaming::onBasicConfigurationComplete() {
  if (mode == IConfigurationCategory::Mode::CreationInPopup || mode == IConfigurationCategory::Mode::CreationInStack) {
    save();
  }
}

void ConfigurationOutputStreaming::onBasicConfigurationCanceled() {
  if (mode == IConfigurationCategory::Mode::CreationInPopup || mode == IConfigurationCategory::Mode::CreationInStack) {
    restore();
  }
}
