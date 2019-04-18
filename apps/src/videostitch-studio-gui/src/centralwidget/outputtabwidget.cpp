// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputtabwidget.hpp"

#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "videostitcher/globalpostprodcontroller.hpp"

#include <QTimer>

OutputTabWidget::OutputTabWidget(QWidget* const parent) : IFreezableWidget("OutputTab", parent), project(nullptr) {
  setupUi(this);
  stackedWidget->setCurrentWidget(pageVideo);
  videoWidget->show();
  videoWidget->setZoomActivated(true);
}

OutputTabWidget::~OutputTabWidget() {
  // clear label
  clearScreenshot();
}

DeviceVideoWidget& OutputTabWidget::getVideoWidget() { return *videoWidget; }

void OutputTabWidget::setProject(ProjectDefinition* p) { project = p; }

void OutputTabWidget::clearProject() { project = nullptr; }

//////////////////////////////////////////

void OutputTabWidget::onUploaderError(const VideoStitch::Status& errorStatus, bool needToExit) {
  if (errorStatus.hasUnderlyingCause(VideoStitch::ErrType::OutOfResources) && project != nullptr) {
    emit reqResetDimensions(project->getPanoConst()->getWidth(), project->getPanoConst()->getHeight(),
                            project->getInputNames());
  } else {
    emit notifyUploadError(errorStatus, needToExit);
  }
}
