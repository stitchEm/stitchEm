// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_outputtabwidget.h"

#include "libvideostitch-base/videowidget.hpp"

#include "libvideostitch-gui/caps/guistatecaps.hpp"
#include "libvideostitch-gui/widgets/stylablewidget.hpp"
#include "libvideostitch-gui/centralwidget/ifreezablewidget.hpp"
#include "libvideostitch-gui/centralwidget/icentraltabwidget.hpp"

class ProjectDefinition;
class SignalCompressionCaps;

class OutputTabWidget : public IFreezableWidget, public ICentralTabWidget, Ui::OutputTabWidget {
  Q_OBJECT
  Q_MAKE_STYLABLE
 public:
  explicit OutputTabWidget(QWidget* const parent = nullptr);
  ~OutputTabWidget();

  DeviceVideoWidget& getVideoWidget();

  virtual bool allowsPlayback() const override { return true; }

 public slots:
  void setProject(ProjectDefinition* p);
  void clearProject();

 signals:
  void refresh(mtime_t date);

  void notifyUploadError(const VideoStitch::Status& status, bool needToExit) const;
  void reqResetDimensions(unsigned panoramaWidth, unsigned panoramaHeight, const QStringList& inputNames);
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

 private slots:
  void onUploaderError(const VideoStitch::Status& errorStatus, bool needToExit);

 private:
  ProjectDefinition* project;
};
