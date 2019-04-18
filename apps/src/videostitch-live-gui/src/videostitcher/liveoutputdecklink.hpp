// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveoutputsdi.hpp"

/**
 * @brief Wrapper for a Decklink output
 */
class LiveOutputDecklink : public LiveOutputSDI {
  Q_OBJECT
 public:
  explicit LiveOutputDecklink(const VideoStitch::Ptv::Value* config,
                              const VideoStitch::OutputFormat::OutputFormatEnum type);
  ~LiveOutputDecklink() = default;

  virtual const QString getOutputDisplayName() const override { return deviceDisplayName; }
  virtual QList<VideoStitch::Audio::SamplingDepth> getSupportedSamplingDepths(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const override;

  VideoStitch::Ptv::Value* serialize() const override;

  bool isConfigurable() const override { return true; }

  OutputConfigurationWidget* createConfigurationWidget(QWidget* const parent) override;
};
