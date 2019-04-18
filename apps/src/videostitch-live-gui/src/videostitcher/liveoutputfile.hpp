// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveoutputfactory.hpp"

#include "libvideostitch-gui/utils/bitratemodeenum.hpp"
#include "libvideostitch-gui/utils/videocodecs.hpp"

/**
 * @brief Wrapper for a file output like mp4, mov
 */
class LiveOutputFile : public LiveWriterFactory {
 public:
  explicit LiveOutputFile(const VideoStitch::Ptv::Value* config,
                          const VideoStitch::Core::PanoDefinition* panoDefinition,
                          const VideoStitch::OutputFormat::OutputFormatEnum type);
  virtual ~LiveOutputFile();

  VideoStitch::Potential<VideoStitch::Output::Output> createWriter(LiveProjectDefinition* project,
                                                                   VideoStitch::FrameRate framerate) override;

  bool isConfigurable() const override { return true; }

  OutputConfigurationWidget* createConfigurationWidget(QWidget* const parent) override;

  virtual const QString getIdentifier() const override;
  virtual const QString getOutputTypeDisplayName() const override { return QStringLiteral("HDD"); }
  virtual const QString getOutputDisplayName() const override;

  QString getFileName() const;
  int getDownsamplingFactor() const;
  VideoStitch::VideoCodec::VideoCodecEnum getCodec() const;
  int getGOP() const;
  int getBFrames() const;
  int getBitRate() const;
  QString getH264Profile() const;
  QString getH264Level() const;
  int getMjpegQualityScale() const;
  QString getProresProfile() const;
  BitRateModeEnum getBitRateMode() const;
  virtual QList<VideoStitch::Audio::SamplingDepth> getSupportedSamplingDepths(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const override;
  PanoSizeChange supportPanoSizeChange(int newWidth, int newHeight) const override;
  QString getPanoSizeChangeDescription(int newWidth, int newHeight) const override;
  void updateForPanoSizeChange(int newWidth, int newHeight) override;

  void setFileName(const QString fileValue);
  void setDownsamplingFactor(const unsigned int dsValue);
  void setGOP(unsigned int gopValue);
  void setBFrames(const unsigned int framesValue);
  void setBitRate(const unsigned int bitrateValue);
  void setCodec(const VideoStitch::VideoCodec::VideoCodecEnum& codecValue);
  void setType(const QString typeValue);
  void setH264Profile(QString newH264Profile);
  void setH264Level(QString newH264Level);
  void setMjpegQualityScale(int newMjpegQualityScale);
  void setProresProfile(QString newProresProfile);
  void setBitRateMode(BitRateModeEnum newBitrateMode);

  virtual VideoStitch::Ptv::Value* serialize() const override;
  virtual QPixmap getIcon() const override;
  virtual QWidget* createStatusWidget(QWidget* const parent) override;

 private:
  void fillOutputValues(const VideoStitch::Ptv::Value* config, const VideoStitch::Core::PanoDefinition* panoDefinition);
  static QString initializeAvStatsLogFile(QString name);

  QString filename;
  int downsampling;
  VideoStitch::VideoCodec::VideoCodecEnum codec;
  QString h264Profile;          // Defined only when codec is H264
  QString h264Level;            // Defined only when codec is H264
  QString h264CommandLineArgs;  // Defined only when codec is H264
  int mjpegQualityScale;        // Defined only when codec is MJPEG
  QString proresProfile;        // Defined only when codec is Prores
  BitRateModeEnum bitrateMode;
  int gop;  // Defined only when codec is H264
  int bframes;
  int bitrate;  // Defined only when codec is H264
};
