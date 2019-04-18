// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <QAudioOutput>

#include <condition_variable>
#include <queue>

class VS_GUI_EXPORT AudioPlayer
    : public QObject,
      public VideoStitch::Output::AudioWriter,
      public VideoStitch::Core::PanoRenderer  // fake renderer to register the video timestamps for synchronization
{
  Q_OBJECT
 public:
  explicit AudioPlayer(QObject* parent = 0);
  virtual ~AudioPlayer();

  QIODevice* device();

  void pushAudio(VideoStitch::Audio::Samples& audioSamples) override;
  std::string getName() const override;
  void render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface>, mtime_t) override;
  void renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t) override;
  void renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t) override;

 public slots:
  void handleStateChanged(QAudio::State);
  void onActivatePlayBack(bool b);

 private:
  void logError(QAudio::Error err);
  static void run(AudioPlayer*);

  std::thread* worker;
  std::mutex mu;
  bool exit = false;
  std::queue<VideoStitch::Audio::Samples> audioQueue;

  std::atomic<mtime_t> currentTimestamp;

  QAudioFormat fmt;
  QAudioOutput* audioOutput;
  QAudioDeviceInfo info;
  QIODevice* dev = nullptr;
  mtime_t delay;
  qreal volume;
};
