// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveoutputfactory.hpp"

#include "libvideostitch-gui/utils/bitratemodeenum.hpp"

#include <QWidget>

#include <memory>

class QHBoxLayout;

class StatusWidget : public QWidget {
  Q_OBJECT
 public:
  explicit StatusWidget(QWidget* const parent);
  ~StatusWidget();

 public slots:
  void connected(const std::string&);
  void connecting();
  void disconnected();

 private:
  QHBoxLayout* layout;
  QLabel* labelStreaming;
  QLabel* labelAnimation;
  QMovie* movieAnimation;
};

/**
 * @brief Wrapper for an rtmp output
 */
class LiveOutputRTMP : public LiveWriterFactory {
 public:
  explicit LiveOutputRTMP(const VideoStitch::Ptv::Value* config,
                          const VideoStitch::Core::PanoDefinition* panoDefinition,
                          const VideoStitch::OutputFormat::OutputFormatEnum type);
  ~LiveOutputRTMP();

  VideoStitch::Potential<VideoStitch::Output::Output> createWriter(LiveProjectDefinition* project,
                                                                   VideoStitch::FrameRate framerate) override;

  bool isConfigurable() const override { return true; }

  OutputConfigurationWidget* createConfigurationWidget(QWidget* const parent) override;

  virtual const QString getIdentifier() const override;
  virtual const QString getOutputDisplayName() const override;

  VideoStitch::Ptv::Value* serialize() const override;
  // add private/hidden parameters that should not be stored within the config file
  VideoStitch::Ptv::Value* serializePrivate() const;

  QString getUrl() const;
  int getDownsamplingFactor() const;
  const BitRateModeEnum getBitRateMode() const;
  int getQualityBalance() const;
  int getGOP() const;
  int getBFrames() const;
  int getBitRate() const;
  int getBufferSize() const;
  int getMinBitrate() const;
  int getTargetUsage() const;
  QString getPubUser() const;
  QString getPubPasswd() const;
  virtual bool showParameters() const;
  virtual bool needsAuthentication() const;
  virtual bool forceConstantBitRate() const;
  QStringList getEncoder() const;
  QList<QStringList> getEncoderList() const;
  QString getPreset() const;
  QString getTune() const;
  QString getProfile() const;
  QString getLevel() const;
  bool cbrPaddingIsEnabled() const;

  virtual QList<VideoStitch::Audio::SamplingDepth> getSupportedSamplingDepths(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const override;
  PanoSizeChange supportPanoSizeChange(int newWidth, int newHeight) const override;
  QString getPanoSizeChangeDescription(int newWidth, int newHeight) const override;
  void updateForPanoSizeChange(int newWidth, int newHeight) override;
  void setUrl(const QString url);
  void setDownsamplingFactor(const unsigned int dsValue);
  void setEncoder(QStringList newEncoder);
  void setPreset(QString newPreset);
  void setTune(QString newTune);
  void setProfile(QString newProfile);
  void setLevel(QString newLevel);
  void setGOP(unsigned int);
  void setBitRateMode(const BitRateModeEnum& mode);
  void setQualityBalance(int newQualityBalance);
  void setBFrames(const unsigned int frames);
  void setBitRate(const unsigned int bitrate);
  void setBufferSize(int newBufferSize);
  void setMinBitrate(int newMinBitrate);
  void setTargetUsage(int);
  void setPubUser(const QString&);
  void setPubPasswd(const QString&);
  void setCbrPaddingEnabled(bool enabled);

  virtual QPixmap getIcon() const override;
  virtual QWidget* createStatusWidget(QWidget* const parent) override;
  virtual bool isAnOutputForAdvancedUser() const;

 private:
  static std::vector<std::string> getSupportedVideoCodecs();
  static QString initializeAvStatsLogFile(QString name);
  void fillOutputValues(const VideoStitch::Ptv::Value* config, const VideoStitch::Core::PanoDefinition* panoDefinition);
  void createEncoderList(QString codec);
  void connectNotifications(VideoStitch::Output::Output& outputWriter);

  QString url;
  int downsampling;
  QStringList encoder;
  QList<QStringList> encoderList;
  QString preset;
  QString tune;
  QString profile;
  QString level;
  int gop;
  int bframes;
  int bitrate;  // Max bitrate when in VBR, bitrate and max bitrate when in CBR
  int bufferSize;
  int targetUsage;  // used by Quick Sync to control the quality/speed tradeoff
  QString memType;  // used by Quick Sync to define the Video Memory
  QString pubUser;
  QString pubPassword;
  BitRateModeEnum bitrateMode;
  int qualityBalance;
  int qp;  // quantization parameter
  bool cbrPaddingEnabled;
  int minBitrate;  // Min bitrate available to handle bandwidth congestion
  int maxBitrate;  // Max bitrate (optional parameter)
  StatusWidget* statusWidget;
};
