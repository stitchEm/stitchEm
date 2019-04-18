// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QFrame>
#include "ui_interactivetabwidget.h"

class OutputControlsPanel;
class QShortcut;
class DeviceInteractiveWidget;

class InteractiveTabWidget : public QFrame, Ui::InteractiveTabWidgetClass {
  Q_OBJECT

 public:
  explicit InteractiveTabWidget(QWidget* const parent = nullptr);
  ~InteractiveTabWidget();

  void setOutputWidgetReference(OutputControlsPanel* buttonsBar);

  OutputControlsPanel* getControlsBar() const;
  DeviceInteractiveWidget* getInteractiveWidget() const;

 public slots:
  void onFullScreenActivated();

 private:
  QShortcut* fullScreenShortCut;
  OutputControlsPanel* outputsPanel;
};
