// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "generic/workflowpage.hpp"

namespace Ui {
class CropWorkflowPage;
}

class SignalCompressionCaps;

class CropWorkflowPage : public WorkflowPage {
  Q_OBJECT

 public:
  explicit CropWorkflowPage(QWidget* parent = nullptr);
  ~CropWorkflowPage();

  void setProject(ProjectDefinition* p) override;
  void initializePage() override;
  void deinitializePage() override;
  void save() override;

 signals:
  void reextract(SignalCompressionCaps* = nullptr);

 private:
  QScopedPointer<Ui::CropWorkflowPage> ui;
};
