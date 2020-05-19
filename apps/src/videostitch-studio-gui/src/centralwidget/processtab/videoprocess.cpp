// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoprocess.hpp"
#include "ui_videoprocess.h"

#include "libvideostitch-gui/utils/videocodecs.hpp"
#include "libvideostitch-gui/centralwidget/formats/format.hpp"
#include "libvideostitch-gui/centralwidget/formats/formatfactory.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/centralwidget/formats/codecs/basicmpegcodec.hpp"
#include "libvideostitch-gui/centralwidget/formats/codecs/trivialcodecs.hpp"
#include "videostitcher/postprodprojectdefinition.hpp"

VideoProcess::VideoProcess(QWidget* const parent)
    : IProcessWidget(parent), ui(new Ui::VideoProcess), currentFormat(nullptr) {
  ui->setupUi(this);
}

VideoProcess::~VideoProcess() {}

void VideoProcess::reactToChangedProject() {
  disconnect(ui->comboFormat, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
             &VideoProcess::onFormatSelected);
  disconnect(ui->comboCodec, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
             &VideoProcess::onCodecSelected);
  // Format and codec
  connect(project, &PostProdProjectDefinition::imagesOrProceduralsOnlyHasChanged, this,
          &VideoProcess::updateFormatComboBox, Qt::UniqueConnection);
  updateFormatComboBox(project->hasImagesOrProceduralsOnly());

  QString outputFormat = project->getOutputVideoFormat();
  QString outputCodec = project->getOutputVideoCodec();
  const int indexFormat = ui->comboFormat->findData(outputFormat);
  if (indexFormat < 0) {
    // If we start with an image or if we remove a video and keep only an image,
    // then the outputFormat can be absent from the combo box
    // so we force the update
    ui->comboFormat->setCurrentIndex(0);
    outputFormat = ui->comboFormat->currentData().toString();
  } else {
    ui->comboFormat->setCurrentIndex(indexFormat);
  }
  currentFormat.reset(FormatFactory::create(outputFormat, this));
  connect(currentFormat.data(), SIGNAL(valueChanged()), this, SLOT(onCodecConfigChanged()));
  updateSupportedCodecs();

  const int indexCodec = ui->comboCodec->findData(outputCodec);
  if (indexCodec < 0) {
    ui->comboCodec->setCurrentIndex(0);
    if (outputCodec == VideoStitch::VideoCodec::getStringFromEnum(VideoStitch::VideoCodec::VideoCodecEnum::MPEG4)) {
      MsgBoxHandler::getInstance()->generic(tr("MPEG4 output codec is not supported anymore."), tr("Warning"),
                                            WARNING_ICON);
      onCodecConfigChanged();
    }
    // The codec can be null when creating a project or if not supported anymore
    outputCodec = ui->comboCodec->currentData().toString();
  } else {
    ui->comboCodec->setCurrentIndex(indexCodec);
  }
  updateCodecConfiguration(outputCodec);

  connect(ui->comboFormat, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
          &VideoProcess::onFormatSelected, Qt::UniqueConnection);
  connect(ui->comboCodec, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
          &VideoProcess::onCodecSelected, Qt::UniqueConnection);
}

void VideoProcess::updateFormatComboBox(bool hasImagesOrProceduralsOnly) {
  const QString oldValue = ui->comboFormat->currentText();
  ui->comboFormat->clear();
  QList<VideoStitch::OutputFormat::OutputFormatEnum> supportedFormats =
      hasImagesOrProceduralsOnly ? VideoStitch::OutputFormat::getSupportedImageFormats()
                                 : VideoStitch::OutputFormat::getSupportedVideoFormats() +
                                       VideoStitch::OutputFormat::getSupportedImageFormats();
  for (auto format : supportedFormats) {
    ui->comboFormat->addItem(VideoStitch::OutputFormat::getDisplayNameFromEnum(format),
                             VideoStitch::OutputFormat::getStringFromEnum(format));
  }
  ui->comboFormat->setCurrentText(oldValue);
}

void VideoProcess::onCodecConfigChanged() { project->setOutputVideoConfig(nullptr, currentFormat->getOutputConfig()); }

void VideoProcess::onCodecSelected(int index) {
  VideoStitch::Ptv::Value* oldConfig = currentFormat ? currentFormat->getOutputConfig() : nullptr;
  const QString data = ui->comboCodec->itemData(index).toString();
  updateCodecConfiguration(data);
  project->setOutputVideoConfig(oldConfig, currentFormat->getOutputConfig());
}

void VideoProcess::onFormatSelected(int index) {
  VideoStitch::Ptv::Value* oldConfig = currentFormat ? currentFormat->getOutputConfig() : nullptr;
  const QString data = ui->comboFormat->itemData(index).toString();
  currentFormat.reset(FormatFactory::create(data, this));
  if (currentFormat == nullptr) {
    return;
  }
  connect(currentFormat.data(), SIGNAL(valueChanged()), this, SLOT(onCodecConfigChanged()));
  updateSupportedCodecs();
  project->setOutputVideoConfig(oldConfig, currentFormat->getOutputConfig());
  emit reqChangeFormat(data);
}

void VideoProcess::updateSupportedCodecs() {
  ui->comboCodec->blockSignals(true);
  const QString oldValue = ui->comboCodec->currentText();
  ui->comboCodec->clear();
  QStringList codecs = currentFormat->getSupportedCodecs();
  codecs.removeOne(VideoStitch::VideoCodec::getStringFromEnum(VideoStitch::VideoCodec::VideoCodecEnum::MPEG4));

  // TODO: not implemented in Studio (CodecFactory, formats/codecs)
  codecs.removeOne(VideoStitch::VideoCodec::getStringFromEnum(VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264));
  codecs.removeOne(VideoStitch::VideoCodec::getStringFromEnum(VideoStitch::VideoCodec::VideoCodecEnum::NVENC_HEVC));

  for (const QString& data : codecs) {
    const VideoStitch::VideoCodec::VideoCodecEnum codec = VideoStitch::VideoCodec::getEnumFromString(data);
    ui->comboCodec->addItem(VideoStitch::VideoCodec::getDisplayNameFromEnum(codec), data);
  }
  ui->comboCodec->setCurrentText(oldValue);
  ui->comboCodec->setVisible(!currentFormat->isACodecToo());
  ui->labelVideoCodec->setVisible(!currentFormat->isACodecToo());
  updateCodecConfiguration(ui->comboCodec->currentData().toString());
  ui->comboCodec->blockSignals(false);
}

void VideoProcess::updateCodecConfiguration(const QString codecName) {
  while (ui->codecLayout->count() > 0) {
    delete ui->codecLayout->takeAt(0);
  }

  if (!currentFormat) {
    return;
  }

  // Some formats are also codecs, we don't want to change them
  if (!codecName.isEmpty()) {
    currentFormat->setCodec(codecName);
  }

  Codec* codec = currentFormat->getCodec();
  if (codec && codec->hasConfiguration()) {
    codec->setFromOutputConfig(project->getOutputConfig()->clone());
    ui->codecLayout->insertWidget(0, codec);
  }
}
