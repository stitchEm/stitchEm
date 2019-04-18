// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "generic/workflowpage.hpp"
#include "videostitcher/livestitchercontroller.hpp"

namespace Ui {
class RigWorkflowPage;
}

class RigWorkflowPage : public WorkflowPage, public VideoStitch::Core::AlgorithmOutput::Listener {
  Q_OBJECT

 public:
  explicit RigWorkflowPage(QWidget* parent = nullptr);
  ~RigWorkflowPage();

  void setProject(ProjectDefinition* p) override;
  void save() override;

  // Reimplemented from VideoStitch::Core::AlgorithmOutput::Listener
  void onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) override;
  void onError(const VideoStitch::Status& error) override;

 signals:
  void reqCalibrate(LiveStitcherController::Callback);
  void reqReset();
  void reqResetPanorama(VideoStitch::Core::PanoramaDefinitionUpdater*, bool saveProject = true);
  void reqSaveProject(QString outputFile, const VideoStitch::Ptv::Value* useless = nullptr);
  void rigChanged();
  void templateSelected(bool selected);
  void useAutoFov(bool autoFov);

  void reqApplyCalibrationImport(const QString path);
  void reqApplyCalibrationTemplate(const QString path);

 private slots:
  void onRigPresetAppliedSuccessfully();
  void onRigPresetFailure();
  void onButtonImportClicked();

 private:
  QScopedPointer<Ui::RigWorkflowPage> ui;
};
