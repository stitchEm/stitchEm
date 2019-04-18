// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QHash>
#include <QWidget>
#include "iprocesswidget.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"

namespace Ui {
class AudioProcess;
}

/**
 * @brief Contains all the output audio configurations (source, encoding, resampling, etc)
 */
class AudioProcess : public IProcessWidget {
  Q_OBJECT

 public:
  explicit AudioProcess(QWidget* const parent = nullptr);
  ~AudioProcess();

 public slots:
  void onFileFormatChanged(const QString format);

 signals:
  void reqChangeAudioInput(const QString name);

 protected:
  virtual void reactToChangedProject() override;
  virtual void reactToClearedProject() override;

 private slots:
  void onCodecChanged(const QString value);
  void onShowAudioConfig(const bool show);
  void onBitrateChanged(const QString value);
  void onAudioInputChanged(const QString value);

 private:
  Ui::AudioProcess* ui;
  QHash<QString, QString> audioFileNames;

  void updateAudioSettings();
  bool isVideo(const QString format) const;
  void setWidgetMode(const QString format);
  void changeCheckState(const Qt::CheckState state);
  void addCodec(const VideoStitch::AudioHelpers::AudioCodecEnum& codec);
  void updateSamplingRate();

  void setModeNoAudio();
  void setModeAudioNotCompatible();
  void setModeAudioNotConfigured();
  void setModeAudioConfigured();
};
