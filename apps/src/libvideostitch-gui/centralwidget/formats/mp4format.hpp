// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "format.hpp"
/**
 * @brief The Mp4Format class represents the MP4 format
 */
class Mp4Format : public Format {
  Q_OBJECT
 public:
  explicit Mp4Format(QWidget* const parent = nullptr);
  ~Mp4Format() = default;
  virtual VideoStitch::Ptv::Value* getOutputConfig() const override;
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override;
  virtual bool isACodecToo() const override;
};
