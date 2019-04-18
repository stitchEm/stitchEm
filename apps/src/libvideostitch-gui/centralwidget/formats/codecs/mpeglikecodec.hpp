// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "basicmpegcodec.hpp"

#include "libvideostitch/frame.hpp"

class QCheckBox;

/**
 * @brief The MpegLikeCodec represent a mpeg like codec class.
 *  Unless the basic class, it also holds a gop attribute, along with a b-frames one
 */
class MpegLikeCodec : public BasicMpegCodec {
  Q_OBJECT

 public:
  explicit MpegLikeCodec(QWidget* const parent = nullptr);
  virtual void setup() override;
  virtual VideoStitch::Ptv::Value* getOutputConfig() const override;
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override;
 public slots:
  void updateGopFromFps(VideoStitch::FrameRate fps);

 protected:
  void addGOPConfiguration();
  void addBFramesConfig();

 protected slots:
  void showAdvancedConfiguration(const bool show);

 private:
  QSpinBox* spinGOP;
  QSpinBox* spinBframes;
  QLabel* labelGOP;
  QLabel* labelBFrames;
  QCheckBox* checkAdvanced;
};
