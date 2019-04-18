// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "mpeglikecodec.hpp"

/**
 * @brief The Mpeg2Codec class represents a widget holding the properties of the mpeg2 codec
 */
class Mpeg2Codec : public MpegLikeCodec {
  Q_OBJECT

 public:
  explicit Mpeg2Codec(QWidget* const parent = nullptr) : MpegLikeCodec(parent) {}

  virtual QString getKey() const override { return QStringLiteral("mpeg2"); }
};
