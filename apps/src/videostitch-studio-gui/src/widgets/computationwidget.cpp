// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "computationwidget.hpp"

#include "videostitcher/postprodprojectdefinition.hpp"

#include "libvideostitch-gui/dialogs/modalprogressdialog.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include "libvideostitch/logging.hpp"

ComputationWidget::ComputationWidget(QWidget* parent)
    : QWidget(parent), algo(nullptr), project(nullptr), algorithmProgressReporterDialog(nullptr) {
  connect(&asyncTaskWatcher, &QFutureWatcher<VideoStitch::Status*>::finished, this,
          &ComputationWidget::finishComputation);
  StateManager::getInstance()->registerObject(this);
}

ComputationWidget::~ComputationWidget() {}

void ComputationWidget::onProjectOpened(ProjectDefinition* p) { project = qobject_cast<PostProdProjectDefinition*>(p); }

void ComputationWidget::clearProject() { project = nullptr; }

// ----------------------- Asynchronous algorithms ---------------------------

void ComputationWidget::startComputationOf(std::function<VideoStitch::Status*()> function) {
  emit reqPause();
  emit reqChangeState(GUIStateCaps::frozen);
  setCursor(Qt::WaitCursor);
  //: Progress dialog title, %0 is the algorithm name
  algorithmProgressReporterDialog.reset(
      new ModalProgressDialog(tr("Processing %0").arg(getAlgorithmName().toLower()), this));
  algorithmProgressReporterDialog->show();
  QFuture<VideoStitch::Status*> asyncTask = QtConcurrent::run(function);
  asyncTaskWatcher.setFuture(asyncTask);
}

void ComputationWidget::finishComputation() {
  algo.reset();
  bool hasBeenCancelled = algorithmProgressReporterDialog->getReporter()->hasBeenCanceled();
  if (hasBeenCancelled) {
    VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Algorithm cancelled" << std::endl;
  }
  algorithmProgressReporterDialog->getReporter()->setValue(100);
  algorithmProgressReporterDialog.reset();
  setCursor(Qt::ArrowCursor);
  emit reqChangeState(GUIStateCaps::stitch);
  manageComputationResult(hasBeenCancelled, asyncTaskWatcher.result());
}

// --------------------- Disable when running -------------------------------

void ComputationWidget::changeState(GUIStateCaps::State s) {
  bool stitchState = s == GUIStateCaps::stitch;
  setEnabled(stitchState && project != nullptr);
}

ProgressReporterWrapper* ComputationWidget::getReporter() const {
  return algorithmProgressReporterDialog ? algorithmProgressReporterDialog->getReporter() : nullptr;
}
