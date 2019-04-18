// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configurationoutputhdd.hpp"
#include "videostitcher/encodedoutputhelpers.hpp"
#include "videostitcher/liveoutputfile.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "guiconstants.hpp"

#include "libvideostitch-gui/utils/h264settingsenum.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include <QFileDialog>

ConfigurationOutputHDD::ConfigurationOutputHDD(LiveOutputFile* output, VideoStitch::OutputFormat::OutputFormatEnum type,
                                               QWidget* const parent)
    : OutputConfigurationWidget(parent), outputRef(output) {
  setupUi(this);
  connect(buttonSelectPath, &QPushButton::clicked, this, &ConfigurationOutputHDD::onButtonBrowseClicked);
  connect(comboReductions, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::updateEncodingSettings);
  connect(comboFormat, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
          &ConfigurationOutputHDD::onFormatChanged);
  connect(comboCodecs, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
          &ConfigurationOutputHDD::onCodecChanged);
  connect(h264ProfileBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::updateBitrateMaximum);

  connect(lineOutputFile, &QLineEdit::textChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(comboReductions, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(comboFormat, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(comboCodecs, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(h264ProfileBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(mjpegQualityScaleSlider, &QSlider::valueChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(proresProfileBox, &QComboBox::currentTextChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);
  connect(spinBitRate, SIGNAL(valueChanged(int)), this, SLOT(onConfigurationChanged()));
  connect(keyframeSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onConfigurationChanged()));
  connect(audioConfig, &ConfigOutputAudio::notifyConfigChanged, this, &ConfigurationOutputHDD::onConfigurationChanged);

  verticalLayout->addLayout(buttonsLayout);
  comboFormat->addItem(
      VideoStitch::OutputFormat::getDisplayNameFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::MP4),
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::MP4));
  comboFormat->addItem(
      VideoStitch::OutputFormat::getDisplayNameFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::MOV),
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::MOV));
  h264ProfileBox->addItem(H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::BASELINE));
  h264ProfileBox->addItem(H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::MAIN));
  h264ProfileBox->addItem(H264Config::ProfileEnum::getDescriptorFromEnum(H264Config::HIGH));
  bitrateModeLabel->setText(BitRateModeEnum::getDescriptorFromEnum(VBR));
  audioConfig->setLiveAudio(outputRef);
  audioConfig->setType(type);
}

ConfigurationOutputHDD::~ConfigurationOutputHDD() {}

void ConfigurationOutputHDD::toggleWidgetState() { configurationWidget->setEnabled(!configurationWidget->isEnabled()); }

LiveOutputFactory* ConfigurationOutputHDD::getOutput() const { return outputRef; }

void ConfigurationOutputHDD::reactToChangedProject() {
  if (projectDefinition) {
    comboReductions->clear();
    for (unsigned int factor : projectDefinition->getDownsampledFactors()) {
      comboReductions->addItem(projectDefinition->getOutputDisplayableString(factor), factor);
    }
    fillWidgetWithValue();
  }
}

void ConfigurationOutputHDD::fillWidgetWithValue() {
  OutputConfigurationWidget::fillWidgetWithValue();

  // General settings

  lineOutputFile->setText(!outputRef->getFileName().isEmpty() ? outputRef->getFileName() : getDefaultOutputFileName());
  comboReductions->setCurrentIndex(comboReductions->findData(outputRef->getDownsamplingFactor()));
  setCurrentIndexFromData(comboFormat, VideoStitch::OutputFormat::getStringFromEnum(outputRef->getType()));
  setCurrentIndexFromData(comboCodecs, VideoStitch::VideoCodec::getStringFromEnum(outputRef->getCodec()));
  updateEncodingSettings();
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  double fps = double(frameRate.num) / double(frameRate.den);
  inputFpsLabel->setText(QString::number(fps));

  // Encoder settings
  h264ProfileBox->setCurrentText(outputRef->getH264Profile());
  if (!outputRef->getH264Level().isEmpty()) {
    h264LevelLabel->setText(outputRef->getH264Level());
    updateBitrateMaximum();
  }
  mjpegQualityScaleSlider->setValue(outputRef->getMjpegQualityScale());
  proresProfileBox->setCurrentText(outputRef->getProresProfile());
  if (outputRef->getBitRate() != -1) {
    spinBitRate->setValue(outputRef->getBitRate());
  }
  keyframeSpinBox->setValue(outputRef->getGOP());

  // Audio settings
  audioConfig->loadParameters();
}

void ConfigurationOutputHDD::onCodecChanged(int index) {
  QString codec = comboCodecs->itemData(index).toString();
  VideoStitch::VideoCodec::VideoCodecEnum codecValue = VideoStitch::VideoCodec::getEnumFromString(codec);
  bool isH264 = (codecValue == VideoStitch::VideoCodec::VideoCodecEnum::H264) ||
                (codecValue == VideoStitch::VideoCodec::VideoCodecEnum::QUICKSYNC_H264) ||
                (codecValue == VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264);
  bool isHEVC = (codecValue == VideoStitch::VideoCodec::VideoCodecEnum::HEVC) ||
                (codecValue == VideoStitch::VideoCodec::VideoCodecEnum::NVENC_HEVC);
  bool isMjpeg = codecValue == VideoStitch::VideoCodec::VideoCodecEnum::MJPEG;
  bool isProres = codecValue == VideoStitch::VideoCodec::VideoCodecEnum::PRORES;

  profileLabel->setVisible(isH264 || isHEVC);
  h264ProfileBox->setVisible(isH264 || isHEVC);
  levelTitleLabel->setVisible(isH264 || isHEVC);
  h264LevelLabel->setVisible(isH264 || isHEVC);
  qualityScaleLabel->setVisible(isMjpeg);
  minQualityLabel->setVisible(isMjpeg);
  maxQualityLabel->setVisible(isMjpeg);
  mjpegQualityScaleSlider->setVisible(isMjpeg);
  proresProfileLabel->setVisible(isProres);
  proresProfileBox->setVisible(isProres);
  bitrateLabel->setVisible(isH264 || isHEVC);
  spinBitRate->setVisible(isH264 || isHEVC);
  keyframeLabel->setVisible(isH264 || isHEVC);
  keyframeSpinBox->setVisible(isH264 || isHEVC);
  updateEncodingSettings();
}

void ConfigurationOutputHDD::onFormatChanged(int index) {
  QString format = comboFormat->itemData(index).toString();
  QString filePath = lineOutputFile->text();
  filePath = filePath.left(filePath.indexOf(".")) + "." + format;
  lineOutputFile->setText(filePath);

  const VideoStitch::OutputFormat::OutputFormatEnum selectedFormat =
      VideoStitch::OutputFormat::getEnumFromString(format);
  QString oldCodec = getCurrentData(comboCodecs);

  comboCodecs->clear();
  QList<VideoStitch::VideoCodec::VideoCodecEnum> codecs =
      VideoStitch::VideoCodec::getSupportedCodecsFor(selectedFormat);
  codecs.removeOne(VideoStitch::VideoCodec::VideoCodecEnum::MPEG4);
  codecs.removeOne(VideoStitch::VideoCodec::VideoCodecEnum::MPEG2);
  if (VideoStitch::GPU::getFramework() != VideoStitch::Discovery::Framework::CUDA) {
    codecs.removeOne(VideoStitch::VideoCodec::VideoCodecEnum::NVENC_H264);
    codecs.removeOne(VideoStitch::VideoCodec::VideoCodecEnum::NVENC_HEVC);
  }
  for (auto codecType : codecs) {
    comboCodecs->addItem(VideoStitch::VideoCodec::getDisplayNameFromEnum(codecType),
                         VideoStitch::VideoCodec::getStringFromEnum(codecType));
  }

  setCurrentIndexFromData(comboCodecs, oldCodec);
}

void ConfigurationOutputHDD::updateEncodingSettings() {
  int factor = comboReductions->currentData().toInt();
  if (factor == 0) {
    return;
  }
  int resizedWidth = int(projectDefinition->getPanoConst()->getWidth()) / factor;
  int resizedHeight = int(projectDefinition->getPanoConst()->getHeight()) / factor;

  auto pixelRate = VideoStitch::getPixelRate(resizedWidth, resizedHeight);
  h264LevelLabel->setText(VideoStitch::getLevelFromMacroblocksRate(
      VideoStitch::getMacroblocksRate(pixelRate),
      VideoStitch::VideoCodec::getEnumFromString(getCurrentData(comboCodecs))));
  updateBitrateMaximum();
  spinBitRate->setValue(VideoStitch::getDefaultBitrate(pixelRate));
}

void ConfigurationOutputHDD::onButtonBrowseClicked() {
  QString path =
      QFileDialog::getSaveFileName(this, tr("Select an output file"), QFileInfo(lineOutputFile->text()).absolutePath(),
                                   "*." + getCurrentData(comboFormat));
  if (!path.isEmpty()) {
    lineOutputFile->setText(QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath()));
  }
}

void ConfigurationOutputHDD::updateBitrateMaximum() {
  QString profile = h264ProfileBox->currentText();
  QString level = h264LevelLabel->text();
  int maxBitrate = VideoStitch::getMaxConstantBitRate(
      profile, level, VideoStitch::VideoCodec::getEnumFromString(getCurrentData(comboCodecs)));
  spinBitRate->setMaximum(maxBitrate);
}

void ConfigurationOutputHDD::saveData() {
  if (outputRef != nullptr) {
    const QString previousFileName = outputRef->getIdentifier();
    QString filePath = lineOutputFile->text();
    const QString format = getCurrentData(comboFormat);
    // remove the ".format" part from the path
    if (filePath.endsWith("." + format, Qt::CaseInsensitive)) {
      filePath = filePath.remove(filePath.length() - format.length() - 1, format.length() + 1);
    }
    outputRef->setFileName(filePath);
    outputRef->setDownsamplingFactor(comboReductions->currentData().toInt());
    outputRef->setType(format);
    outputRef->setCodec(VideoStitch::VideoCodec::getEnumFromString(getCurrentData(comboCodecs)));
    if (h264ProfileBox->isVisible()) {
      outputRef->setH264Profile(h264ProfileBox->currentText());
    }
    if (h264LevelLabel->isVisible()) {
      outputRef->setH264Level(h264LevelLabel->text());
    }
    if (mjpegQualityScaleSlider->isVisible()) {
      outputRef->setMjpegQualityScale(mjpegQualityScaleSlider->value());
    }
    if (proresProfileBox->isVisible()) {
      outputRef->setProresProfile(proresProfileBox->currentText());
    }
    // Bitrate mode is always VBR
    outputRef->setBitRateMode(BitRateModeEnum(BitRateMode::VBR));
    outputRef->setBitRate(spinBitRate->isVisible() ? spinBitRate->value() : -1);
    outputRef->setGOP(keyframeSpinBox->isVisible() ? keyframeSpinBox->value() : -1);
    audioConfig->saveConfiguration();

    projectDefinition->updateOutputId(previousFileName);
    emit reqChangeOutputId(previousFileName, outputRef->getIdentifier());
  }
}

QString ConfigurationOutputHDD::getCurrentData(QComboBox* combo) const { return combo->currentData().toString(); }

void ConfigurationOutputHDD::setCurrentIndexFromData(QComboBox* combo, const QString& format) {
  const int index = combo->findData(format);
  if (index < 0) {
    combo->setCurrentIndex(0);
  } else {
    combo->setCurrentIndex(index);
  }
}
