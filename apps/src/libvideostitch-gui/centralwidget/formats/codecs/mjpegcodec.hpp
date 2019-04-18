// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "basicmpegcodec.hpp"

/**
 * @brief The MjpegCodec class represents a widget holding the properties of the MPJEG codec
 */
class MjpegCodec : public Codec {
  Q_OBJECT

 public:
  explicit MjpegCodec(QWidget* const parent = nullptr);
  virtual QString getKey() const override;
  virtual void setup() override;
  virtual bool hasConfiguration() const override { return true; }
  virtual VideoStitch::Ptv::Value* getOutputConfig() const override;
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override;

 private:
  QGridLayout* mainLayout;
  QSlider* sliderScale;
  QLabel* labelQuality;
  QLabel* labelTitle;
};
