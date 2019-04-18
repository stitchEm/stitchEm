// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropinputcontroller.hpp"

#include "generic/backgroundcontainer.hpp"
#include "generic/genericdialog.hpp"
#include "outputcontrolspanel.hpp"
#include "videostitcher/globallivecontroller.hpp"
#include "widgetsmanager.hpp"

#include "libvideostitch-gui/widgets/crop/cropwindow.hpp"

static const int WIDGET_EXTENSION(200);

CropInputController::CropInputController(OutputControlsPanel* panel, QObject* const parent)
    : QObject(parent), project(nullptr) {
  connect(panel->buttonCropInputs, &QPushButton::clicked, this, &CropInputController::onShowCropWindow);
}

void CropInputController::onShowCropWindow() {
  if (project != nullptr) {
    CropWindow cropWindow(project, project->getProjectLensType(), WIDGET_EXTENSION);
    StitcherController* stitcherController = GlobalController::getInstance().getController();
    connect(&cropWindow.getCropWidget(), &CropWidget::reextract, stitcherController,
            &StitcherController::reextractOnce);
    connect(&cropWindow.getCropWidget(), &CropWidget::reqRegisterRender, stitcherController,
            &StitcherController::registerSourceRender);
    connect(&cropWindow.getCropWidget(), &CropWidget::reqUnregisterRender, stitcherController,
            &StitcherController::unregisterSourceRender, Qt::BlockingQueuedConnection);
    connect(&cropWindow.getCropWidget(), &CropWidget::reqApplyCrops, this, &CropInputController::applyCropsAsked);
    cropWindow.getCropWidget().initializeTabs();
    BackgroundContainer* container = new BackgroundContainer(&cropWindow, tr("Crop inputs"),
                                                             WidgetsManager::getInstance()->getMainWindowRef(), false);
    connect(&cropWindow, &CropWindow::finished, this, [=]() {
      container->hide();
      container->deleteLater();
    });
    container->show();
    cropWindow.exec();
    cropWindow.getCropWidget().deinitializeTabs();
  }
}

void CropInputController::setProject(ProjectDefinition* p) { project = p; }

void CropInputController::onCropApplied() {
  WidgetsManager::getInstance()->closeLoadingDialog();

  GenericDialog* errorDialog =
      new GenericDialog(tr("Crop inputs"), tr("Inputs cropped successfully"), GenericDialog::DialogMode::ACCEPT,
                        WidgetsManager::getInstance()->getMainWindowRef());
  errorDialog->show();
}

void CropInputController::applyCropsAsked(const QVector<Crop>& crops, const InputLensClass::LensType lensType) {
  LiveStitcherController* stitcherController = GlobalLiveController::getInstance().getController();
  connect(stitcherController, &LiveStitcherController::notifyInputsCropped, this, [=]() {
    onCropApplied();
    disconnect(stitcherController, &LiveStitcherController::notifyInputsCropped, this, nullptr);
  });

  emit reqApplyCrops(crops, lensType);
}
