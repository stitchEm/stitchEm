// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"

#include <QDropEvent>
#include <QFrame>

namespace Ui {
class WelcomeScreenWidget;
}

class SoftwareHelpWidget;
class ProjectSelectionWidget;

class VS_GUI_EXPORT WelcomeScreenWidget : public QFrame {
  Q_OBJECT

 public:
  explicit WelcomeScreenWidget(QWidget* parent = nullptr);
  ~WelcomeScreenWidget();
  void updateContent();

 signals:
  void notifyProjectOpened();
  void notifyNewProject();
  void notifyProjectSelected(const QString names);
  void notifyFilesDropped(QDropEvent* e);

 protected:
  virtual void resizeEvent(QResizeEvent* event) override;

 private:
  void setSmallSizeOrder();
  void setBigSizeOrder();
  void setLogoSize();
  void addWidgets();
  float getLogoRatio() const;
  Ui::WelcomeScreenWidget* ui;
  SoftwareHelpWidget* helpWidget;
  ProjectSelectionWidget* projectWidget;
};
