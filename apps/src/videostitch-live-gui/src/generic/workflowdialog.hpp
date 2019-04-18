// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QDialog>

class ProjectDefinition;
class WorkflowPage;
namespace Ui {
class WorkflowDialog;
}

class WorkflowDialog : public QDialog {
  Q_OBJECT

 public:
  explicit WorkflowDialog(QWidget* parent = nullptr);
  ~WorkflowDialog();

  ProjectDefinition* getProject() const;

  void setProject(ProjectDefinition* p);
  void addPage(WorkflowPage* page);  // Takes ownership of the page
  void completeCurrentPage(QString message);
  void showErrorMessage(QString errorMessage);
  void hideErrorMessage();
  void showWaitingPage(QString message);
  void closeWaitingPage();
  void blockNextStep(bool block);

 private:
  WorkflowPage* getCurrentPage() const;
  int getCurrentPageIndex() const;

 private slots:
  void saveCurrentPage();
  void goBack();

 private:
  QScopedPointer<Ui::WorkflowDialog> ui;
  ProjectDefinition* project;
  QVector<QSharedPointer<WorkflowPage>> pages;
  QScopedPointer<QPushButton> buttonBack;
  QScopedPointer<QPushButton> buttonApply;
};
