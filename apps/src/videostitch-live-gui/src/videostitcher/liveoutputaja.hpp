// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEOUTPUTAJA_HPP
#define LIVEOUTPUTAJA_HPP

#include "liveoutputsdi.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch/audio.hpp"

class LiveOutputAJA : public LiveOutputSDI {
  Q_OBJECT
 public:
  explicit LiveOutputAJA(const VideoStitch::Ptv::Value* config, const VideoStitch::OutputFormat::OutputFormatEnum type);

  virtual const QString getOutputDisplayName() const override { return deviceDisplayName; }
  virtual QList<VideoStitch::Audio::SamplingDepth> getSupportedSamplingDepths(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const override;
  bool getAudioIsEnabled() const;

  virtual VideoStitch::Ptv::Value* serialize() const override;

  bool isConfigurable() const override { return true; }

  OutputConfigurationWidget* createConfigurationWidget(QWidget* const parent) override;
  PanoSizeChange supportPanoSizeChange(int newWidth, int newHeight) const override;
  QString getPanoSizeChangeDescription(int newWidth, int newHeight) const override;
  void updateForPanoSizeChange(int newWidth, int newHeight) override;

  void setAudioIsEnabled(bool audio);

 private:
  void fillOutputValues(const VideoStitch::Ptv::Value* config);
  QString toReadableName(const QString name) const;
  int nameToDevice(const QString name) const;
  int nameToChannel(const QString name) const;

 private:
  int device;
  int channel;
  bool audioIsEnabled = false;
};

#endif  // LIVEOUTPUTAJA_HPP
