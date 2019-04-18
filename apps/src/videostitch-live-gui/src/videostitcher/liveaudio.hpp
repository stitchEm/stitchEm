// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEAUDIO_HPP
#define LIVEAUDIO_HPP

#include "libvideostitch/ptv.hpp"
#include <QObject>

class LiveOutputFactory;

class LiveAudio : public QObject {
  Q_OBJECT
 public:
  LiveAudio();
  ~LiveAudio();

  void initialize(const VideoStitch::Ptv::Value* config, const LiveOutputFactory& output);

  const QString getAudioCodec() const;
  int getSamplingRate() const;
  int getBitrate() const;
  const QString getSamplingFormat() const;
  const QString getChannelLayout() const;
  void serializeIn(VideoStitch::Ptv::Value* config) const;

  void setAudioCodec(const QString codecValue);
  void setSamplingRate(const int samplingRateValue);
  void setSamplingFormat(const QString samplingDepthValue);
  void setChannelLayout(const QString channelLayoutValue);
  void setBitrate(const int bitrateValue);

  /**
   * @brief Check if the audio output is defined (codec, ...)
   */
  bool isConfigured() const;

 private:
  int samplingRate;
  int bitrate;
  QString samplingFormat;
  QString audioCodec;
  QString channelLayout;
};

#endif  // LIVEAUDIO_HPP
