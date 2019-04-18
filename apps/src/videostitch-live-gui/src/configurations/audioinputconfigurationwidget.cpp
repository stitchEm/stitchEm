// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioinputconfigurationwidget.hpp"
#include "ui_audioinputconfigurationwidget.h"

#include "plugin/pluginscontroller.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"

static int TYPE_ROLE = Qt::UserRole;
static int ID_ROLE = Qt::UserRole + 1;

AudioInputConfigurationWidget::AudioInputConfigurationWidget(QWidget* parent)
    : IConfigurationCategory(parent), ui(new Ui::AudioInputConfigurationWidget) {
  ui->setupUi(this);
  ui->titleLabel->setProperty("vs-title1", true);
  ui->verticalLayout->addLayout(buttonsLayout);

  connect(ui->deviceBox, &QComboBox::currentTextChanged, this, &AudioInputConfigurationWidget::updateVisibility);
  connect(ui->deviceBox, &QComboBox::currentTextChanged, this, &AudioInputConfigurationWidget::onConfigurationChanged);
  connect(ui->nbChannelsBox, &QComboBox::currentTextChanged, this,
          &AudioInputConfigurationWidget::onConfigurationChanged);
  connect(ui->rateBox, &QComboBox::currentTextChanged, this, &AudioInputConfigurationWidget::onConfigurationChanged);
}

AudioInputConfigurationWidget::~AudioInputConfigurationWidget() {}

void AudioInputConfigurationWidget::setPluginsController(const PluginsController* newPluginsController) {
  pluginsController = newPluginsController;
}

AudioConfiguration AudioInputConfigurationWidget::getAudioConfiguration() const {
  AudioConfiguration audioConfig;
  if (ui->deviceBox->currentText() != getTextForNoDevice()) {
    audioConfig.inputName = ui->deviceBox->currentData(ID_ROLE).toString().toStdString();
    const QString type = ui->deviceBox->currentData(TYPE_ROLE).toString();
    audioConfig.type = type.toStdString();
    if (type != VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA)) {
      audioConfig.nbAudioChannels = ui->nbChannelsBox->currentText().toInt();
      audioConfig.audioRate = VideoStitch::Audio::SamplingRate(ui->rateBox->currentData().toInt());
    }
  }
  return audioConfig;
}

void AudioInputConfigurationWidget::reactToChangedProject() {
  setAudioConfiguration(projectDefinition->getAudioConfiguration());
}

void AudioInputConfigurationWidget::setAudioConfiguration(const AudioConfiguration& config) {
  disconnect(ui->deviceBox, &QComboBox::currentTextChanged, this,
             &AudioInputConfigurationWidget::updateDeviceSupportedValues);

  // Update device box
  QMap<QString, QPair<QString, QString>> audioDevices;  // <display name, <type, id>>
  const QString portAudioType =
      VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::PORTAUDIO);
  for (QString portAudioDevice : pluginsController->listInputDeviceNames(portAudioType)) {
    audioDevices.insert(portAudioDevice, qMakePair(portAudioType, portAudioDevice));
  }
  if (VSSettings::getSettings() && VSSettings::getSettings()->getShowExperimentalFeatures()) {
    audioDevices.insert(
        getTextForProcedural(),
        qMakePair(QString::fromStdString(VideoStitch::Audio::getAudioGeneratorId()), getTextForProcedural()));
  }
  if (projectDefinition->getVideoInputType() == VideoStitch::InputFormat::InputFormatEnum::AJA) {
    const QString ajaType = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
    VideoStitch::Plugin::VSDiscoveryPlugin* ajaPlugin = PluginsController::getPluginByName(ajaType);
    for (QString ajaDeviceName : projectDefinition->getVideoInputNames()) {
      const VideoStitch::Plugin::DiscoveryDevice ajaDevice =
          PluginsController::getDeviceByName(ajaPlugin, ajaDeviceName);
      audioDevices.insert(QString::fromStdString(ajaDevice.displayName), qMakePair(ajaType, ajaDeviceName));
    }
  }

  ui->deviceBox->clear();
  ui->deviceBox->addItem(getTextForNoDevice());  // Should be the first item
  int itemIndex = 1;
  for (QString audioDevice : audioDevices.keys()) {
    ui->deviceBox->addItem(audioDevice);
    ui->deviceBox->setItemData(itemIndex, audioDevices.value(audioDevice).first, TYPE_ROLE);
    ui->deviceBox->setItemData(itemIndex, audioDevices.value(audioDevice).second, ID_ROLE);
    ++itemIndex;
  }
  if (config.isValid()) {  // If not valid, the first item (no device) will be selected
    const int index = ui->deviceBox->findData(QString::fromStdString(config.inputName), ID_ROLE);
    ui->deviceBox->setCurrentIndex(index);
  }

  // Update sampling rate box & nb channels box
  updateDeviceSupportedValues();
  ui->nbChannelsBox->setCurrentText(QString::number(config.nbAudioChannels));
  ui->rateBox->setCurrentText(VideoStitch::AudioHelpers::getSampleRateString(config.audioRate));

  connect(ui->deviceBox, &QComboBox::currentTextChanged, this,
          &AudioInputConfigurationWidget::updateDeviceSupportedValues);
}

void AudioInputConfigurationWidget::updateVisibility(QString device) {
  const bool noDevice = device == getTextForNoDevice();
  const bool ajaDevice = ui->deviceBox->currentData(TYPE_ROLE).toString() ==
                         VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
  ui->rateLabel->setVisible(!noDevice && !ajaDevice);
  ui->rateBox->setVisible(!noDevice && !ajaDevice);
  ui->nbChannelsLabel->setVisible(!noDevice && !ajaDevice);
  ui->nbChannelsBox->setVisible(!noDevice && !ajaDevice);
}

QString AudioInputConfigurationWidget::getTextForNoDevice() const {
  //: No device
  return tr("None");
}

QString AudioInputConfigurationWidget::getTextForProcedural() const {
  return VideoStitch::InputFormat::getDisplayNameFromEnum(VideoStitch::InputFormat::InputFormatEnum::AUDIOPROCEDURAL);
}

void AudioInputConfigurationWidget::updateDeviceSupportedValues() {
  Q_ASSERT(pluginsController != nullptr);

  const QString currentAudioInputName = ui->deviceBox->currentData(ID_ROLE).toString();
  const QString currentAudioInputType = ui->deviceBox->currentData(TYPE_ROLE).toString();
  const QString oldNbChannels = ui->nbChannelsBox->currentText();
  const QString oldRate = ui->rateBox->currentText();

  QVector<VideoStitch::Audio::SamplingRate> audioRates;
  QStringList nbAudioChannels;
  if (currentAudioInputName == getTextForProcedural()) {
    audioRates =
        QVector<VideoStitch::Audio::SamplingRate>::fromStdVector(VideoStitch::AudioHelpers::samplingRatesSupported);
    for (int nbInputChannel : VideoStitch::AudioHelpers::nbInputChannelsSupported) {
      nbAudioChannels.append(QString::number(nbInputChannel));
    }
  } else {
    audioRates = pluginsController->listAudioSamplingRates(currentAudioInputType, currentAudioInputName);
    nbAudioChannels = pluginsController->listNbAudioChannels(currentAudioInputType, currentAudioInputName);
  }

  // Update sampling rate box
  ui->rateBox->clear();
  for (auto rate : audioRates) {
    ui->rateBox->addItem(VideoStitch::AudioHelpers::getSampleRateString(rate), int(rate));
  }
  ui->rateBox->setCurrentText(oldRate);

  // Update nb channels box
  ui->nbChannelsBox->clear();
  ui->nbChannelsBox->addItems(nbAudioChannels);
  ui->nbChannelsBox->setCurrentText(oldNbChannels);
}
