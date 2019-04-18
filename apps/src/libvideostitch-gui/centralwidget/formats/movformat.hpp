// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "format.hpp"
/**
 * @brief The MovFormat class represents the mov format
 */
class MovFormat : public Format {
  Q_OBJECT
 public:
  explicit MovFormat(QWidget* const parent = nullptr);
  ~MovFormat() = default;
  virtual VideoStitch::Ptv::Value* getOutputConfig() const override;
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override;
  virtual bool isACodecToo() const override;
};
