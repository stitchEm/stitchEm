// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>

class ProjectDefinition;
class WorkflowDialog;

class WorkflowPage : public QWidget {
  Q_OBJECT

 public:
  explicit WorkflowPage(QWidget* parent = nullptr);
  ~WorkflowPage() = default;

  virtual void setProject(ProjectDefinition* p) = 0;
  virtual void initializePage() {}
  virtual void deinitializePage() {}
  virtual void save() = 0;
  void setWorkflowDialog(WorkflowDialog* w);

 protected:
  WorkflowDialog* workflowDialog;
};
