// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_sourcestabwidget.h"
#include "calibration/calibrationactioncontroller.hpp"
#include "libvideostitch-gui/centralwidget/sourcewidget.hpp"
#include "libvideostitch-gui/caps/guistatecaps.hpp"
#include "libvideostitch/stereoRigDef.hpp"

class IConfigurationCategory;
class LiveProjectDefinition;

class SourcesTabWidget : public QWidget, public Ui::SourcesTabWidgetClass, public GUIStateCaps {
  Q_OBJECT
  Q_MAKE_STYLABLE

 public:
  explicit SourcesTabWidget(QWidget* const parent = nullptr);
  ~SourcesTabWidget();

  SourceWidget* getSourcesWidget() const;

  void restore();
  void showLoadingWidget();
  void showWidgetEdition(IConfigurationCategory* configurationWidget);

 public slots:
  virtual void changeState(GUIStateCaps::State state) override;
  void setProject(ProjectDefinition* project);
  void clearProject();
  void onFileHasTobeSaved();
  void onButtonSnapshotClicked();
  void onButtonConfigure3DRigClicked();
  void onRigConfigurationSuccess();

 signals:
  void reqChangeState(GUIStateCaps::State s) override;
  void reqStitcherReload();
  void notifyTakeSourcesSnapshot(const QString& path);
  void reqSaveProject(QString, const VideoStitch::Ptv::Value* = nullptr);
  void notifyRigConfigured(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                           const VideoStitch::Core::StereoRigDefinition::Geometry geometry, const double diameter,
                           const double ipd, const QVector<int> leftInputs, const QVector<int> rightInputs);

 private:
  void activateOptions(bool activate);

 private:
  SourceWidget* sourceWidget;
  LiveProjectDefinition* projectDefinition;
};
