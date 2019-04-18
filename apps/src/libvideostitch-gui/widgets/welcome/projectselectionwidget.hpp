// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QFrame>

namespace Ui {
class ProjectSelectionWidget;
}

class ProjectSelectionWidget : public QFrame {
  Q_OBJECT

 public:
  explicit ProjectSelectionWidget(QWidget* parent = nullptr);
  ~ProjectSelectionWidget();

 public slots:
  void onContentUpdated();

 signals:
  void notifyProjectOpened();
  void notifyNewProject();
  void notifyProjectSelected(const QString& name);
  void notifyFilesDropped(QDropEvent* e);

 protected:
  virtual bool eventFilter(QObject* watched, QEvent* event) override;

 private slots:
  void onProjectSelected(const QModelIndex& index);
  void onSampleSelected(const QModelIndex& index);

 private:
  void loadRecentProjects();
  Ui::ProjectSelectionWidget* ui;
};
