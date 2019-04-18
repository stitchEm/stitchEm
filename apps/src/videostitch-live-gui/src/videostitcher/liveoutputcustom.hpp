// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveoutputfactory.hpp"

#include "libvideostitch/parse.hpp"

#include <memory>

class LiveOutputCustom : public LiveOutputFactory {
 public:
  explicit LiveOutputCustom(const VideoStitch::Ptv::Value* config,
                            const VideoStitch::OutputFormat::OutputFormatEnum outputType);
  virtual ~LiveOutputCustom() {}

  virtual const QString getIdentifier() const override;
  virtual const QString getOutputTypeDisplayName() const override;
  virtual const QString getOutputDisplayName() const override;

  VideoStitch::Ptv::Value* serialize() const override;
  virtual QPixmap getIcon() const override;
  virtual QWidget* createStatusWidget(QWidget* const parent) override;

 private:
  std::unique_ptr<VideoStitch::Ptv::Value> config;
  QString customType;
};
