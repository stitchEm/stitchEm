// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "mpeglikecodec.hpp"

/**
 * @brief The Mpeg4Codec class represents a widget holding the properties of the mpeg4 codec
 */
class Mpeg4Codec : public MpegLikeCodec {
  Q_OBJECT

 public:
  explicit Mpeg4Codec(QWidget *const parent = nullptr);
  virtual void setup() override;
  virtual QString getKey() const override;
  virtual bool meetsSizeRequirements(int width, int height) const override;
  virtual void correctSizeToMeetRequirements(int &width, int &height) override;
};
