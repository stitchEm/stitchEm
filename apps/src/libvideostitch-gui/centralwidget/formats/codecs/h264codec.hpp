// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "mpeglikecodec.hpp"

static const int H264_MAX_WIDTH(4096);

/**
 * @brief The H264Codec class represents a widget holding the properties of the h264 codec
 */
class H264Codec : public MpegLikeCodec {
  Q_OBJECT

 public:
  explicit H264Codec(QWidget *const parent = nullptr);
  virtual QString getKey() const override;
  virtual void correctSizeToMeetRequirements(int &width, int &height) override;
  virtual bool meetsSizeRequirements(int width, int height) const override;
};
