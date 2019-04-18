// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#include "ui_interactivetabwidget.h"

#include "libvideostitch-gui/centralwidget/ifreezablewidget.hpp"
#include "libvideostitch-gui/centralwidget/icentraltabwidget.hpp"

#include <QPointer>

class SignalCompressionCaps;

class InteractiveTabWidget : public IFreezableWidget, public ICentralTabWidget, Ui::InteractiveTabWidgetClass {
  Q_OBJECT
 public:
  explicit InteractiveTabWidget(QWidget* const parent = nullptr);
  ~InteractiveTabWidget();
  DeviceInteractiveWidget* getInteractiveWidget();
  virtual bool allowsPlayback() const override { return true; }

 signals:
  void reqRestitch(SignalCompressionCaps* = nullptr);

 protected slots:
  void clearScreenshot() override {}

 protected:
  // ifreezablewidget state machine methods
  // deactivated until able to unload/reload opengl
  void freeze() override {}
  void unfreeze() override {}
  void showGLView() override {}
  void connectToDeviceWriter() override {}
};
