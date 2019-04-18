// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationactioncontroller.hpp"

#include "calibration/calibrationworkflowpage.hpp"
#include "calibration/cropworkflowpage.hpp"
#include "calibration/rigworkflowpage.hpp"
#include "generic/backgroundcontainer.hpp"
#include "generic/genericdialog.hpp"
#include "generic/workflowdialog.hpp"
#include "widgetsmanager.hpp"
#include "outputcontrolspanel.hpp"

#include "libvideostitch-gui/mainwindow/vssettings.hpp"

CalibrationActionController::CalibrationActionController(OutputControlsPanel* widget)
    : QObject(), outputControlsPanel(widget), projectDefinition(nullptr) {
  outputControlsPanel->buttonCalibrationToggleControlPoints->setVisible(false);  // temp, VSA-4525
  outputControlsPanel->labelCalibrationToggleControlPoints->setVisible(false);   // temp, VSA-4525

  connect(outputControlsPanel, &OutputControlsPanel::notifyNewCalibration, this,
          &CalibrationActionController::onLaunchCalibrationWorkflow);
  connect(outputControlsPanel->buttonCalibrationClear, &QPushButton::clicked, this,
          &CalibrationActionController::onCalibrationClear);
  connect(outputControlsPanel->buttonCalibrationImprove, &QPushButton::clicked, this,
          &CalibrationActionController::onLaunchCalibrationWorkflow);
}

CalibrationActionController::~CalibrationActionController() {}

void CalibrationActionController::setProject(ProjectDefinition* proj) {
  projectDefinition = proj;
  if (projectDefinition->getPanoConst().get()) {
    configureButtonsAfterCalibration();
  }
}

void CalibrationActionController::clearProject() { projectDefinition = nullptr; }

void CalibrationActionController::onCalibrationImportError(QString message,
                                                           const VideoStitch::Status& calibrationImportStatus) {
  if (calibrationImportStatus.ok()) {
    GenericDialog::createAcceptDialog(tr("Calibration import"), message,
                                      WidgetsManager::getInstance()->getMainWindowRef());
  } else {
    GenericDialog* calibrationErrorDialog =
        new GenericDialog(message, calibrationImportStatus, WidgetsManager::getInstance()->getMainWindowRef());
    calibrationErrorDialog->show();
  }
}

void CalibrationActionController::onLaunchCalibrationWorkflow() {
  const bool improveMode = sender() == outputControlsPanel->buttonCalibrationImprove;

  WorkflowDialog dialog;
  dialog.setProject(projectDefinition);
  CalibrationWorkflowPage* calibrationWorkflowPage = new CalibrationWorkflowPage();
  connect(calibrationWorkflowPage, &CalibrationWorkflowPage::calibrationChanged, this,
          &CalibrationActionController::configureButtonsAfterCalibration);
  if (!improveMode) {
    RigWorkflowPage* rigWorkflowPage = new RigWorkflowPage();
    connect(rigWorkflowPage, &RigWorkflowPage::reqApplyCalibrationImport, this,
            &CalibrationActionController::reqApplyCalibrationImport);
    connect(rigWorkflowPage, &RigWorkflowPage::reqApplyCalibrationTemplate, this,
            &CalibrationActionController::reqApplyCalibrationTemplate);
    dialog.addPage(rigWorkflowPage);
    dialog.addPage(new CropWorkflowPage());
    connect(rigWorkflowPage, &RigWorkflowPage::useAutoFov, calibrationWorkflowPage,
            &CalibrationWorkflowPage::setAutoFov);
  }
  dialog.addPage(calibrationWorkflowPage);

  BackgroundContainer* container = new BackgroundContainer(&dialog, tr("Calibration workflow"),
                                                           WidgetsManager::getInstance()->getMainWindowRef(), false);
  connect(&dialog, &WorkflowDialog::finished, this, [container]() {
    container->hide();
    container->deleteLater();
  });

  container->show();
  dialog.exec();
}

void CalibrationActionController::onCalibrationClear() {
  outputControlsPanel->showMainTab();
  emit reqClearCalibration();
}

void CalibrationActionController::configureButtonsAfterCalibration() {
  const bool hasCP = projectDefinition->getPanoConst()->hasCalibrationControlPoints();
  outputControlsPanel->buttonCropInputs->setVisible(true);
  outputControlsPanel->labelCropInputs->setVisible(true);
  outputControlsPanel->buttonCalibrationImprove->setVisible(hasCP);
  outputControlsPanel->labelCalibrationImprove->setVisible(hasCP);
  outputControlsPanel->buttonCalibrationClear->setVisible(true);
  outputControlsPanel->labelCalibrationClear->setVisible(true);
  outputControlsPanel->buttonCalibrationAdapt->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  outputControlsPanel->labelCalibrationAdapt->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
}
