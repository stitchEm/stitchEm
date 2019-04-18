// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropworkflowpage.hpp"
#include "ui_cropworkflowpage.h"

#include "generic/workflowdialog.hpp"

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

CropWorkflowPage::CropWorkflowPage(QWidget* parent) : WorkflowPage(parent), ui(new Ui::CropWorkflowPage) {
  ui->setupUi(this);
  static const int WIDGET_EXTENSION(200);
  ui->cropWidget->setWidgetExtension(WIDGET_EXTENSION);
}

CropWorkflowPage::~CropWorkflowPage() { deinitializePage(); }

void CropWorkflowPage::setProject(ProjectDefinition* p) {
  ui->cropWidget->setProject(p);
  StitcherController* stitcherController = GlobalController::getInstance().getController();
  connect(ui->cropWidget, &CropWidget::reextract, stitcherController, &StitcherController::reextractOnce);
  connect(ui->cropWidget, &CropWidget::reqRegisterRender, stitcherController,
          &StitcherController::registerSourceRender);
  connect(ui->cropWidget, &CropWidget::reqUnregisterRender, stitcherController,
          &StitcherController::unregisterSourceRender, Qt::BlockingQueuedConnection);
  connect(ui->cropWidget, &CropWidget::reqApplyCrops, stitcherController, &StitcherController::applyCrops);
}

void CropWorkflowPage::initializePage() {
  ui->cropWidget->setLensType(workflowDialog->getProject()->getProjectLensType());
  ui->cropWidget->initializeTabs();
  ui->cropWidget->setDefaultCrop();
}

void CropWorkflowPage::deinitializePage() { ui->cropWidget->deinitializeTabs(); }

void CropWorkflowPage::save() {
  ui->cropWidget->applyCrop();
  workflowDialog->completeCurrentPage(tr("Inputs cropped"));
}
