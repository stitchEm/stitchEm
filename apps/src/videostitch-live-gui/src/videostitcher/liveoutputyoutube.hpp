// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEOUTPUTYOUTUBE_HPP
#define LIVEOUTPUTYOUTUBE_HPP

#include "liveoutputrtmp.hpp"

class LiveOutputYoutube : public LiveOutputRTMP {
  Q_OBJECT
 public:
  explicit LiveOutputYoutube(const VideoStitch::Ptv::Value* config,
                             const VideoStitch::Core::PanoDefinition* panoDefinition,
                             const VideoStitch::OutputFormat::OutputFormatEnum type);

  VideoStitch::Ptv::Value* serialize() const override;

  virtual bool showParameters() const override;
  virtual bool needsAuthentication() const override;
  virtual bool forceConstantBitRate() const override;

  virtual QPixmap getIcon() const override;
  virtual bool checkIfIsActivable(const VideoStitch::Core::PanoDefinition* panoDefinition,
                                  QString& message) const override;
  virtual bool isAnOutputForAdvancedUser() const override;

  // VSA-6594
#if !defined(YOUTUBE_OUTPUT)
  virtual bool isSupported() const override { return false; }
#endif

  QString getBroadcastId() const;
  void setBroadcastId(const QString& value);

 protected:
  void initializeAudioOutput(const VideoStitch::Ptv::Value* config) const override;

 private:
  void fillOutputValues(const VideoStitch::Ptv::Value* config, const VideoStitch::Core::PanoDefinition* panoDefinition);

  QString broadcastId;
};

#endif  // LIVEOUTPUTYOUTUBE_HPP
