// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QFrame>
#include "libvideostitch-gui/caps/guistatecaps.hpp"
#include "libvideostitch-gui/centralwidget/icentraltabwidget.hpp"
#include "libvideostitch/config.hpp"

#ifdef Q_OS_WIN
#include <ShObjIdl.h>
#endif

namespace Ui {
class ProcessTabWidget;
}

class SignalCompressionCaps;
class OutputFileProcess;
class VideoProcess;
class AudioProcess;
class IProcessWidget;
class ProjectDefinition;
class PostProdProjectDefinition;
class QSpacerItem;

/**
 * @brief Tabn containing all the output configurations
 */
class ProcessTabWidget : public QFrame, public ICentralTabWidget {
  Q_OBJECT
 public:
  explicit ProcessTabWidget(QWidget* const parent = nullptr);
  ~ProcessTabWidget();

 public slots:
  void onProjectOpened(ProjectDefinition* project);
  void clearProject();
  void processOutput(const frameid_t first, const frameid_t last);
  void changeState(GUIStateCaps::State) {}

 signals:
  void reqProjectOpened(PostProdProjectDefinition* project);
  void reqProjectCleared();
  void reqSendToBatch(const bool copy);
  void reqSavePtv();
  void panoSizeChanged(const unsigned width, const unsigned height);
  void reqChangeAudioInput(const QString name);
  void reqChangeState(GUIStateCaps::State s);
  void reqUpdateSequence(const QString start, const QString end);
  void reqReset(SignalCompressionCaps* compressor);

#ifdef Q_OS_WIN
  void reqStartProgress();
  void reqStopProgress();
  void reqChangeProgressValue(quint64 current, quint64 total);
  void reqChangeProgressState(TBPFLAG state);
#endif

 private:
  void addProcessWidget(IProcessWidget* const processWidget);
  QStringList getGPUs() const;
  Ui::ProcessTabWidget* ui;
  OutputFileProcess* outputFileProcessWidget;
  VideoProcess* videoProcessWidget;
  AudioProcess* audioProcessWidget;
};
