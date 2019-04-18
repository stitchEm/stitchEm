// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_outputtabwidget.h"

#include "exposure/exposureactioncontroller.hpp"
#include "outputcontrolspanel.hpp"
#include "configurations/configoutputswidget.hpp"
#include "videostitcher/livestitchercontroller.hpp"

#include "libvideostitch-base/projection.hpp"
#include "libvideostitch/ptv.hpp"

#include <QFrame>
#include <QMap>

class QShortcut;
class OutPutTabWidget : public QFrame, public Ui::OutPutTabWidgetClass {
  Q_OBJECT

 public:
  explicit OutPutTabWidget(QWidget* const parent = nullptr);
  ~OutPutTabWidget();

  void setOutputWidgetReference(OutputControlsPanel* controlBar);
  void toggleOutput(const QString& id);
  void setOutputActionable(const QString& id, bool actionable);
  void updateOutputId(const QString oldName, const QString newName);
  void restore();

  DeviceVideoWidget* getVideoWidget() const;
  OutputControlsPanel* getControlsBar() const;
  ConfigPanoramaWidget* getConfigPanoramaWidget() const;
  AudioProcessorsWidget* getAudioProcessorWidget() const;

 private:
  OutputControlsPanel* outputControls;
  LiveProjectDefinition* projectDefinition;
  QShortcut* fullScreenShortCut;

 public slots:
  void onButtonSnapshotClicked();
  void setProject(ProjectDefinition* project);
  void clearProject();
  void onFileHasTobeSaved();
  void onFullScreenActivated();
  void onInvalidPano();

 signals:
  void reqResetPanorama(VideoStitch::Core::PanoramaDefinitionUpdater*, bool save = true);
  void reqStitcherReload(SignalCompressionCaps* comp = nullptr);
  void notifyOutputActivated(const QString& id);
  void reqSwitchProjection(VideoStitch::Projection, double);
  void notifyTakePanoSnapshot(const QString&);
  void reqSaveProject(QString, const VideoStitch::Ptv::Value* = nullptr);
  void refresh(mtime_t timestamp);

 private slots:
  void reSetupVideoWidget();
  void showVideoWidgetPage();
  void showPanoramaEditionPage();
  void showAudioProcessorsPage();
};
