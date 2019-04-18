// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "format.hpp"

class SimpleFormat : public Format {
 public:
  explicit SimpleFormat(QString theFormat, ExtensionHandler* theHandler, QWidget* const parent = nullptr);
  ~SimpleFormat() = default;

  VideoStitch::Ptv::Value* getOutputConfig() const override;
  bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override;
  bool isACodecToo() const override;

 private:
  QString format;
};
