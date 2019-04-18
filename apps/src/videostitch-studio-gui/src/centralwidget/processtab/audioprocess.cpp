// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioprocess.hpp"
#include "ui_audioprocess.h"

#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "videostitcher/postprodprojectdefinition.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

AudioProcess::AudioProcess(QWidget* const parent) : IProcessWidget(parent), ui(new Ui::AudioProcess) {
  ui->setupUi(this);
  ui->comboAudioBitRate->addItems(VideoStitch::AudioHelpers::getAudioBitrates());
  ui->stackedWidget->setCurrentWidget(ui->pageAudio);
  addCodec(VideoStitch::AudioHelpers::AudioCodecEnum::AAC);
  addCodec(VideoStitch::AudioHelpers::AudioCodecEnum::MP3);
  connect(ui->groupAudio, &QGroupBox::toggled, this, &AudioProcess::onShowAudioConfig);
  connect(ui->comboAudioTrack, &QComboBox::currentTextChanged, this, &AudioProcess::onAudioInputChanged);
  connect(ui->comboAudioCodec, static_cast<void (QComboBox::*)(const QString&)>(&QComboBox::activated), this,
          &AudioProcess::onCodecChanged);
  connect(ui->comboAudioBitRate, static_cast<void (QComboBox::*)(const QString&)>(&QComboBox::activated), this,
          &AudioProcess::onBitrateChanged);
}

AudioProcess::~AudioProcess() { delete ui; }

void AudioProcess::reactToChangedProject() {
  ui->comboAudioTrack->blockSignals(true);
  ui->comboAudioTrack->clear();
  ui->comboAudioBitRate->setCurrentText(QString::number(project->getOutputAudioBitrate()));
  const int index = ui->comboAudioCodec->findData(project->getOutputAudioCodec());
  if (index < 0) {
    ui->comboAudioCodec->setCurrentIndex(0);
  } else {
    ui->comboAudioCodec->setCurrentIndex(index);
  }
  ui->comboAudioTrack->setCurrentIndex(0);
  QStringList audioInputs = project->getAudioInputNames();
  audioFileNames.clear();
  for (QString& audioInput : audioInputs) {
    const QString& audioInputShortName = QFileInfo(audioInput).fileName();
    audioFileNames[audioInputShortName] = audioInput;
    ui->comboAudioTrack->addItem(audioInputShortName);
  }
  const QString& selectedAudioShortName =
      QFileInfo(QString(project->getAudioPipeConst()->getSelectedAudio().c_str())).fileName();
  ui->comboAudioTrack->setCurrentText(selectedAudioShortName);
  ui->comboAudioTrack->blockSignals(false);
  ui->labelChannelsValue->setText(project->getOutputAudioChannels());
  updateSamplingRate();
  setWidgetMode(project->getOutputVideoFormat());
}

void AudioProcess::reactToClearedProject() {}

void AudioProcess::onCodecChanged(const QString value) {
  Q_UNUSED(value)
  updateAudioSettings();
  updateSamplingRate();
}

void AudioProcess::onShowAudioConfig(const bool /* show */) { updateAudioSettings(); }

void AudioProcess::onBitrateChanged(const QString /* value */) { updateAudioSettings(); }

void AudioProcess::onAudioInputChanged(const QString value) {
  Q_UNUSED(value);
  emit reqChangeAudioInput(audioFileNames[ui->comboAudioTrack->currentText()]);
  updateAudioSettings();
}

void AudioProcess::onFileFormatChanged(const QString format) {
  if (project) {
    setWidgetMode(format);
    updateAudioSettings();
  }
}

void AudioProcess::updateAudioSettings() {
  // TODO: these are default value. This should be fixed once the new audio pipeline is finished
  if (ui->groupAudio->isChecked() && isVisible()) {
    project->setOutputAudioConfig(ui->comboAudioCodec->currentData().toString(),
                                  ui->comboAudioBitRate->currentText().toInt(),
                                  audioFileNames[ui->comboAudioTrack->currentText()]);
  } else {
    project->removeAudioSource();
  }
}

void AudioProcess::updateSamplingRate() {
  const QString codec = ui->comboAudioCodec->currentData().toString();
  const int samplingRate =
      VideoStitch::AudioHelpers::getDefaultSamplingRate(VideoStitch::AudioHelpers::getCodecFromString(codec));
  ui->labelSampleRateValue->setText(QString::number(samplingRate) + tr(" Hz"));
}

bool AudioProcess::isVideo(const QString format) const {
  return VideoStitch::OutputFormat::isVideoFormat(VideoStitch::OutputFormat::getEnumFromString(format));
}

void AudioProcess::setWidgetMode(const QString format) {
  const bool isAVideo = isVideo(format);
  const bool hasAudio = GlobalController::getInstance().getController()->hasInputAudio();
  const bool hasAudioConfig = project->hasAudioConfiguration();
  // No video format (image or invalid one)
  if (!isAVideo) {
    setModeNoAudio();
  }
  // Video format but no audio readers
  if (isAVideo && !hasAudio) {
    setModeAudioNotCompatible();
  }
  // Video format with no audio configuration
  if (isAVideo && hasAudio && !hasAudioConfig) {
    setModeAudioNotConfigured();
  }
  // Video format with audio configuration
  if (isAVideo && hasAudio && hasAudioConfig) {
    setModeAudioConfigured();
  }
}

void AudioProcess::changeCheckState(const Qt::CheckState state) {
  ui->groupAudio->blockSignals(true);
  ui->groupAudio->setChecked(state == Qt::CheckState::Checked);
  ui->groupAudio->blockSignals(false);
}

void AudioProcess::addCodec(const VideoStitch::AudioHelpers::AudioCodecEnum& codec) {
  const QString name = VideoStitch::AudioHelpers::getDisplayNameFromCodec(codec);
  const QString data = VideoStitch::AudioHelpers::getStringFromCodec(codec);
  ui->comboAudioCodec->addItem(name, data);
}

void AudioProcess::setModeNoAudio() {
  setVisible(false);
  changeCheckState(Qt::CheckState::Unchecked);
}

void AudioProcess::setModeAudioNotCompatible() {
  setVisible(true);
  ui->stackedWidget->setCurrentWidget(ui->pageWarning);
  ui->labelWarningMessage->setText(tr("No compatible audio found in the inputs"));
  changeCheckState(Qt::CheckState::Unchecked);
}

void AudioProcess::setModeAudioNotConfigured() {
  setVisible(true);
  ui->stackedWidget->setCurrentWidget(ui->pageAudio);
  changeCheckState(Qt::CheckState::Unchecked);
}

void AudioProcess::setModeAudioConfigured() {
  setVisible(true);
  ui->stackedWidget->setCurrentWidget(ui->pageAudio);
  changeCheckState(Qt::CheckState::Checked);
}
