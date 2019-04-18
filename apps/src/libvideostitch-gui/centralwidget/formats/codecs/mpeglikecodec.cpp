// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mpeglikecodec.hpp"

#include "libvideostitch/parse.hpp"
#include <QCheckBox>

MpegLikeCodec::MpegLikeCodec(QWidget* const parent)
    : BasicMpegCodec(parent),
      spinGOP(nullptr),
      spinBframes(nullptr),
      labelGOP(nullptr),
      labelBFrames(nullptr),
      checkAdvanced(new QCheckBox(tr("Advanced settings"), this)) {
  checkAdvanced->setObjectName("showAdvanced");
  checkAdvanced->setProperty("vs-advanced-box", true);
  checkAdvanced->setChecked(false);
  checkAdvanced->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
  mainLayout->addWidget(checkAdvanced, 2, 0);
  checkAdvanced->setAttribute(Qt::WA_LayoutUsesWidgetRect);
  connect(checkAdvanced, &QCheckBox::toggled, this, &MpegLikeCodec::showAdvancedConfiguration);
}

void MpegLikeCodec::setup() {
  BasicMpegCodec::setup();
  addGOPConfiguration();
  addBFramesConfig();
  showAdvancedConfiguration(false);
}

VideoStitch::Ptv::Value* MpegLikeCodec::getOutputConfig() const {
  VideoStitch::Ptv::Value* outputConfig = BasicMpegCodec::getOutputConfig();
  if (spinGOP) {
    outputConfig->get("gop")->asInt() = spinGOP->value();
  }
  if (spinBframes) {
    outputConfig->get("b_frames")->asInt() = spinBframes->value();
  }
  return outputConfig;
}

bool MpegLikeCodec::setFromOutputConfig(const VideoStitch::Ptv::Value* config) {
  if (!BasicMpegCodec::setFromOutputConfig(config)) {
    return false;
  }
  int gopSize = 0;
  int b_frames = 0;

  if (VideoStitch::Parse::populateInt("Ptv", *config, "b_frames", b_frames, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    gopSize = LIBAV_WRITER_DEFAULT_GOP_SIZE;
  }

  if (VideoStitch::Parse::populateInt("Ptv", *config, "gop", gopSize, false) != VideoStitch::Parse::PopulateResult_Ok) {
    b_frames = LIBAV_WRITER_DEFAULT_B_FRAMES;
  }

  if (spinGOP) {
    spinGOP->blockSignals(true);
    spinGOP->setValue(gopSize);
    spinGOP->blockSignals(false);
  }

  if (spinBframes) {
    spinBframes->blockSignals(true);
    spinBframes->setValue(b_frames);
    spinBframes->blockSignals(false);
  }

  return true;
}

void MpegLikeCodec::updateGopFromFps(VideoStitch::FrameRate fps) {
  spinBframes->setValue(std::ceil(fps.num / fps.den));
}

void MpegLikeCodec::addGOPConfiguration() {
  spinGOP = new QSpinBox(this);
  spinGOP->setFixedSize(CONTROL_WIDTH, CONTROL_HEIGHT);
  spinGOP->setValue(LIBAV_WRITER_DEFAULT_GOP_SIZE);
  spinGOP->setMinimum(1);
  spinGOP->setMaximum(LIBAV_WRITER_MAX_GOP_SIZE);
  spinGOP->setFocusPolicy(Qt::StrongFocus);
  labelGOP = new QLabel(tr("GOP:"), this);
  labelGOP->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
  mainLayout->addWidget(labelGOP, 3, 0);
  mainLayout->addWidget(spinGOP, 3, 1);
  connect(spinGOP, SIGNAL(valueChanged(int)), this, SIGNAL(valueChanged()));
}

void MpegLikeCodec::addBFramesConfig() {
  labelBFrames = new QLabel(tr("B-frames:"), this);
  labelBFrames->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
  spinBframes = new QSpinBox(this);
  spinBframes->setFixedSize(CONTROL_WIDTH, CONTROL_HEIGHT);
  spinBframes->setValue(LIBAV_WRITER_DEFAULT_B_FRAMES);
  spinBframes->setMaximum(LIBAV_WRITER_MAX_B_FRAMES);
  spinBframes->setFocusPolicy(Qt::StrongFocus);
  mainLayout->addWidget(labelBFrames, 4, 0);
  mainLayout->addWidget(spinBframes, 4, 1);
  connect(spinBframes, SIGNAL(valueChanged(int)), this, SIGNAL(valueChanged()));
}

void MpegLikeCodec::showAdvancedConfiguration(const bool show) {
  if (labelBFrames) {
    labelBFrames->setVisible(show);
    spinBframes->setVisible(show);
  }
  if (labelGOP) {
    labelGOP->setVisible(show);
    spinGOP->setVisible(show);
  }
}
