// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sourcewidget.hpp"
#include "ui_sourcewidget.h"

#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/utils/sourcewidgetlayoututil.hpp"

SourceWidget::SourceWidget(const bool drops, QWidget* parent)
    : IFreezableWidget("Source", parent), ui(new Ui::SourceWidget), project(nullptr) {
  ui->setupUi(this);
  setAcceptDrops(drops);

#ifdef HIDE_MASK_FEATURE
  ui->bottomSource->hide();
#endif
  viewPtr = std::shared_ptr<MultiVideoWidget>(ui->widgetMain);
}

SourceWidget::~SourceWidget() { delete ui; }

MultiVideoWidget& SourceWidget::getMultiVideoWidget() { return *ui->widgetMain; }

// ----------------------- Open/Close Project ---------------------------------

void SourceWidget::createThumbnails(
    std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string>> inputs,
    std::vector<std::shared_ptr<VideoStitch::Core::SourceRenderer>>* renderers) {
  // remove the previous thumbnails
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  videoStitcher->lockedFunction([inputs, renderers, this]() -> void {
    clearThumbnails();
    for (unsigned int i = 0; i < inputs.size(); ++i) {
      renderers->push_back(viewPtr);
    }
  });
}

void SourceWidget::clearThumbnails() { viewPtr->clearTextures(); }

int SourceWidget::getNumbSources() const {
  if (viewPtr) {
    return viewPtr->getNumbTextures();
  } else {
    return 0;
  }
}

void SourceWidget::setProject(ProjectDefinition* p) {
  project = p;
  connect(project,
          static_cast<void (ProjectDefinition::*)(int, unsigned char*, int, int)>(&ProjectDefinition::reqUpdateMask),
          this, static_cast<void (SourceWidget::*)(int, unsigned char*, int, int)>(&SourceWidget::updateMask),
          Qt::UniqueConnection);

  connect(project, static_cast<void (ProjectDefinition::*)(int, QImage*)>(&ProjectDefinition::reqUpdateMask), this,
          static_cast<void (SourceWidget::*)(int, QImage*)>(&SourceWidget::updateMask), Qt::UniqueConnection);

  connect(this, &SourceWidget::reqUpdateMasks, project, &ProjectDefinition::updateMasks, Qt::UniqueConnection);
}

void SourceWidget::clearProject() {
  project = nullptr;
  clearThumbnails();
}

void SourceWidget::changeState(GUIStateCaps::State s) {
  IFreezableWidget::changeState(s);
#ifndef HIDE_MASK_FEATURE
  if (s == GUIStateCaps::stitch) {
    ui->bottomSource->show();
  }
#endif
}

// ------------------------------ Masks management -------------------------------------

void SourceWidget::maskToggled(bool toggled) {
  if (toggled) {
    emit reqUpdateMasks();
  }
}

void SourceWidget::updateMask(int /* index */, unsigned char* /* maskData */, int /* width */, int /* height */) {}

// We don't need to delete the mask since the QPixmap generated from the image will take the ownership
void SourceWidget::updateMask(int /* index */, QImage* /* mask */) {}

void SourceWidget::onUploaderError(const VideoStitch::Status& errorStatus, bool needToExit) {
  if (errorStatus.hasUnderlyingCause(VideoStitch::ErrType::OutOfResources) && project != nullptr) {
    emit reqResetDimensions(project->getPanoConst()->getWidth(), project->getPanoConst()->getHeight(),
                            project->getInputNames());
  } else {
    emit notifyUploadError(errorStatus, needToExit);
  }
}
