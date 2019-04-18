// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mjpegcodec.hpp"
#include "libvideostitch/parse.hpp"

MjpegCodec::MjpegCodec(QWidget *const parent)
    : Codec(parent),
      mainLayout(new QGridLayout(this)),
      sliderScale(new QSlider(Qt::Horizontal, this)),
      labelQuality(new QLabel(tr("Quality / Compression"), this)),
      labelTitle(new QLabel(tr("Quality scale:"), this)) {}

QString MjpegCodec::getKey() const { return QStringLiteral("mjpeg"); }

void MjpegCodec::setup() {
  mainLayout->setContentsMargins(0, 0, 0, 0);
  mainLayout->setSpacing(CONTROLS_SPACING);
  sliderScale->setMinimum(LIBAV_WRITER_MIN_QSCALE);
  sliderScale->setMaximum(LIBAV_WRITER_MAX_QSCALE);
  sliderScale->setSingleStep(1);
  sliderScale->setFixedWidth(CONTROL_WIDTH);
  labelTitle->setFixedWidth(LABEL_WIDTH);
  mainLayout->addWidget(labelTitle, 0, 0);
  mainLayout->addWidget(sliderScale, 0, 1);
  mainLayout->addWidget(labelQuality, 0, 2);
  setLayout(mainLayout);
  connect(sliderScale, &QSlider::sliderReleased, this, [=]() { emit valueChanged(); });
}

VideoStitch::Ptv::Value *MjpegCodec::getOutputConfig() const {
  VideoStitch::Ptv::Value *outputConfig = VideoStitch::Ptv::Value::emptyObject();
  outputConfig->get("scale")->asInt() = sliderScale->value();
  return outputConfig;
}

bool MjpegCodec::setFromOutputConfig(const VideoStitch::Ptv::Value *config) {
  int scale = LIBAV_WRITER_MIN_QSCALE;
  if (VideoStitch::Parse::populateInt("Ptv", *config, "scale", scale, false) != VideoStitch::Parse::PopulateResult_Ok) {
    return false;
  }
  sliderScale->setValue(scale);
  return true;
}
