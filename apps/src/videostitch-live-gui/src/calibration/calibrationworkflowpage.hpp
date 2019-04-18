// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "generic/workflowpage.hpp"
#include "videostitcher/livestitchercontroller.hpp"

#include "libvideostitch/ptv.hpp"

#include <memory>

namespace Ui {
class CalibrationWorkflowPage;
}

class CalibrationWorkflowPage : public WorkflowPage, public VideoStitch::Core::AlgorithmOutput::Listener {
  Q_OBJECT

 public:
  explicit CalibrationWorkflowPage(QWidget* parent = nullptr);
  ~CalibrationWorkflowPage();

  void setProject(ProjectDefinition* p) override;
  void initializePage() override;
  void deinitializePage() override;
  void save() override;

  // Reimplemented from VideoStitch::Core::AlgorithmOutput::Listener
  void onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) override;
  void onError(const VideoStitch::Status& error) override;

 public slots:
  void setAutoFov(bool tempUseAutoFov);

 signals:
  void reqCalibrate(LiveStitcherController::Callback);
  void reqResetPanorama(VideoStitch::Core::PanoramaDefinitionUpdater*, bool saveProject = true);
  void calibrationChanged();

 private:
  void calibrate();
  std::unique_ptr<VideoStitch::Ptv::Value> buildCalibrationConfig() const;

 private slots:
  void onCalibrationSuccess();
  void onCalibrationFailure();

 private:
  QScopedPointer<Ui::CalibrationWorkflowPage> ui;
  ProjectDefinition* project;
  int oldFrameAction;
  bool useAutoFov = false;
};
