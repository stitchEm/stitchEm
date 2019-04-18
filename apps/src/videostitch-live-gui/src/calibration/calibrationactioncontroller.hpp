// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

#include <QObject>

class OutputControlsPanel;
class ProjectDefinition;

class CalibrationActionController : public QObject {
  Q_OBJECT
 public:
  explicit CalibrationActionController(OutputControlsPanel* widget);
  ~CalibrationActionController();

 public slots:
  void onLaunchCalibrationWorkflow();
  void onCalibrationClear();
  void onCalibrationImportError(QString message, const VideoStitch::Status& status);
  void setProject(ProjectDefinition*);
  void clearProject();

 signals:
  void reqClearCalibration();
  void reqApplyCalibrationImport(const QString& path);
  void reqApplyCalibrationTemplate(const QString& path);

  friend class ProjectWorkWidget;

 private slots:
  void configureButtonsAfterCalibration();

 private:
  OutputControlsPanel* outputControlsPanel;
  ProjectDefinition* projectDefinition;
};
