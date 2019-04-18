// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "codec.hpp"
#include "libvideostitch/parse.hpp"
#include <QGridLayout>
#include <QSpinBox>
#include <QLabel>

static const unsigned JPEG_WRITER_DEFAULT_QUALITY(90);
static const unsigned JPEG_QUALITY_MIN_VALUE(1);
static const unsigned JPEG_QUALITY_MAX_VALUE(100);

/**
 * @brief The JpegCodec class represents a widget holding the properties of the jpeg codec
 */
class JpegCodec : public Codec {
  Q_OBJECT

 public:
  explicit JpegCodec(QWidget* const parent = nullptr)
      : Codec(parent),
        mainLayout(new QGridLayout(this)),
        sliderQuality(new QSlider(Qt::Horizontal, this)),
        labelValue(new QLabel(this)),
        labelQuality(new QLabel(tr("Quality:"), this)) {
    sliderQuality->setMinimum(JPEG_QUALITY_MIN_VALUE);
    sliderQuality->setMaximum(JPEG_QUALITY_MAX_VALUE);
    setContentsMargins(0, 0, 0, 0);
    connect(sliderQuality, &QSlider::valueChanged, this,
            [=](int value) { labelValue->setText(QString::number(value)); });
    connect(sliderQuality, &QSlider::sliderReleased, this, [=]() { emit valueChanged(); });
  }

  virtual void setup() override {
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(CONTROLS_SPACING);
    labelQuality->setFixedWidth(LABEL_WIDTH);
    sliderQuality->setFixedWidth(CONTROL_WIDTH);
    labelValue->setFixedWidth(CONTROL_WIDTH - 2 * CONTROLS_SPACING);
    sliderQuality->setValue(JPEG_WRITER_DEFAULT_QUALITY);
    mainLayout->addWidget(labelQuality, 0, 0);
    mainLayout->addWidget(sliderQuality, 0, 1);
    mainLayout->addWidget(labelValue, 0, 2);
    setLayout(mainLayout);
  }

  virtual bool hasConfiguration() const override { return true; }

  virtual VideoStitch::Ptv::Value* getOutputConfig() const override {
    VideoStitch::Ptv::Value* outputConfig = VideoStitch::Ptv::Value::emptyObject();
    outputConfig->get("quality")->asInt() = sliderQuality->value();
    return outputConfig;
  }

  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override {
    int quality = JPEG_QUALITY_MIN_VALUE;
    if (VideoStitch::Parse::populateInt("Ptv", *config, "quality", quality, false) !=
        VideoStitch::Parse::PopulateResult_Ok) {
      return false;
    }
    sliderQuality->setValue(quality);
    labelQuality->setText(QString::number(quality));
    return true;
  }

  virtual QString getKey() const override { return QStringLiteral("jpg"); }

 private:
  QGridLayout* mainLayout;
  QSlider* sliderQuality;
  QLabel* labelValue;
  QLabel* labelQuality;
};
