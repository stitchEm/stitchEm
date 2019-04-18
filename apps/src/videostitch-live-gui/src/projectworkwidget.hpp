// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#include "ui_projectworkwidget.h"

#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/packet.hpp"
#include "libvideostitch-gui/mainwindow/frameratecompute.hpp"
#include "libvideostitch-gui/mainwindow/gpuinfoupdater.hpp"

#include "libvideostitch-base/vslog.hpp"

class AudioPlayer;
class CalibrationActionController;
class CalibrationUpdateController;
class CropInputController;
class ExposureActionController;
class OutputsController;
class OutputPluginController;
class InputPluginController;

class ProjectWorkWidget : public QWidget, public Ui::ProjectWorkWidgetClass, public GUIStateCaps {
  Q_OBJECT

 public:
  explicit ProjectWorkWidget(QWidget* const parent = nullptr);
  ~ProjectWorkWidget();

  void startApplication();
  void openFile(QVector<int> devices, const QFileInfo& fileName, const int customWidth = 0, const int customHeight = 0);
  void startNewProject(QVector<int> devices, const QString& name);

  bool outputIsActivated() const;
  bool algorithmIsActivated() const;

 public slots:
  /**
   * @brief This function will synchronously close the project, delete it, ask to delete the stitcher controller
   * and clear all the GUI
   */
  void onCloseProject();
  void saveProject();

 signals:
  void reqChangeState(GUIStateCaps::State s);
  void reqRefresh(qint64);
  void reqCreateNewProject();
  void reqOpenVAHFile(QString inFile, int customWidth, int customHeight);
  void reqThreadQuit();
  void reqForceGPUInfoRefresh();
  void reqReset();
  void reqSave(const QString& file, const VideoStitch::Ptv::Value* = nullptr);
  void notifyProjectClosed();
  void reqReopenProject(const QString& file);
  void notifyProjectOpened();
  void notifyPanoResized(const unsigned customWidth, const unsigned customHeight);
  void reqResetDimensions(const unsigned customWidth, const unsigned customHeight);
  void notifyRemoveAudio();
  void notifyAudioPlaybackActivated(bool);
  // Kernel compile
  void reqCancelKernelCompile();
  void notifyBackendCompileProgress(const QString& message, double progress);
  void notifyBackendCompileDone();

  void reqDisableWindow();
  void reqEnableWindow();

 protected slots:
  virtual void changeState(GUIStateCaps::State s) { Q_UNUSED(s) }

 private:
  void initializeMainTab();
  void initializeStitcher();
  void registerWidgetConnections();  // Register connections that are independent of the stitcher controller
  void registerStitcherSignals();
  void registerOutputSignals();
  void registerLoggers();
  void registerSignalInjections();
  void registerControllers();
  void registerMetaTypes();
  void createStitcherController(QVector<int> devices);
  void startOpenGL();
  void stopOpenGL();

 private slots:
  void registerRenderer(std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>* renderers);
  void setProject(ProjectDefinition* project);
  void setNeedToExtract(bool newNeedToExtract);
  void onPlay();
  void onPause();
  void onCleanStitcher();
  void onStitcherErrorMessage(const VideoStitch::Status& status, bool needToExit);
  void onEndOfStreamReached();
  void onAudioLoadError(const QString& title, const QString& message);
  void updateTopBar(mtime_t date);
  /**
   * @brief This slot will update the widgets enability after the activation (or deactivation) of an output or of an
   * automatic algorithm.
   */
  void updateEnabilityAfterActivation();
  void updateClockForTab(int tab);
  void updateNextFrameAction();
  void onStitcherReset();

 private:
  LiveProjectDefinition* projectDefinition;
  FramerateCompute framerateComputer;
  GPUInfoUpdater gpuInfoUpdater;
  OutputsController* outputsController;
  CalibrationActionController* calibrationActionController;
  CalibrationUpdateController* calibrationUpdateController;
  ExposureActionController* exposureController;
  InputPluginController* inputPluginController;
  OutputPluginController* outputPluginController;
  QScopedPointer<CropInputController> cropInputController;
  QThread stitcherControllerThread;
  StitcherController::NextFrameAction nextFrameAction;
  int needToExtract = 0;
};
