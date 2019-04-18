// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "codec.hpp"
#include <QVBoxLayout>
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QSlider>
#include <QGridLayout>

// Common MPEG encoders values
static const unsigned LIBAV_WRITER_DEFAULT_BITRATE(15000000);
static const unsigned LIBAV_WRITER_MAX_MP4_BITRATE(110000000);
static const unsigned LIBAV_WRITER_MIN_MP4_BITRATE(300000);
static const char* LIBAV_WRITER_DEFAULT_BITRATE_MODE("VBR");
static const char* LIBAV_WRITER_DEFAULT_CONTAINER("mp4");
static const float LIBAV_WRITER_DEFAULT_FRAMERATE(25.f);
static const unsigned LIBAV_WRITER_DEFAULT_GOP_SIZE((int)(1.0 * LIBAV_WRITER_DEFAULT_FRAMERATE));
static const unsigned LIBAV_WRITER_MAX_GOP_SIZE((int)(10.0 * LIBAV_WRITER_DEFAULT_FRAMERATE));
static const unsigned LIBAV_WRITER_DEFAULT_B_FRAMES(2);
static const unsigned LIBAV_WRITER_MAX_B_FRAMES(5);
static const unsigned LIBAV_WRITER_MIN_QSCALE(1);
static const unsigned LIBAV_WRITER_MAX_QSCALE(31);
static const unsigned LIBAV_DEFAULT_AUDIO_BITRATE(128000);

// Widget values
static const unsigned LABEL_WIDTH(150);
static const unsigned CONTROL_WIDTH(200);
static const unsigned CONTROL_HEIGHT(23);
static const unsigned CONTROLS_SPACING(9);

/**
 * @brief The BasicMpegCodec represent a basic codec class holding a bitrate and a bitrate mode attribute.
 */
class BasicMpegCodec : public Codec {
  Q_OBJECT
 public:
  explicit BasicMpegCodec(QWidget* parent = nullptr);
  virtual void setup() override;
  virtual bool hasConfiguration() const override;
  virtual VideoStitch::Ptv::Value* getOutputConfig() const override;
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override;

 protected:
  QGridLayout* mainLayout;
  QComboBox* bitrateModeComboBox;
  QSlider* bitrateSlider;
  QLabel* bitrateLabel;
  QLabel* bitrateModeLabel;
  QLineEdit* bitrateLineEdit;
 public slots:
  void updateTextEdit(int value);
  void updatePosition(const QString str);
};
