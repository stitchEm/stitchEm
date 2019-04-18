// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "photometriccalibrationwidget.hpp"
#include "ui_photometriccalibrationwidget.h"

PhotometricCalibrationWidget::PhotometricCalibrationWidget(QWidget* const parent)
    : QWidget(parent), ui(new Ui::PhotometricCalibrationWidget), renderVignette(false) {
  ui->setupUi(this);
  ui->tabCentral->setCurrentWidget(ui->tabVignette);
  connect(ui->buttonRenderVignette, &QPushButton::clicked, this,
          &PhotometricCalibrationWidget::onRenderVignetteClicked);
}

PhotometricCalibrationWidget::~PhotometricCalibrationWidget() {}

void PhotometricCalibrationWidget::setEmorValues(double emor1, double emor2, double emor3, double emor4, double emor5) {
  ui->cameraResponseRenderWidget->setEmorParams(emor1, emor2, emor3, emor4, emor5);
  ui->labelEmorA->setText(QString::number(emor1));
  ui->labelEmorB->setText(QString::number(emor2));
  ui->labelEmorC->setText(QString::number(emor3));
  ui->labelEmorD->setText(QString::number(emor4));
  ui->labelEmorE->setText(QString::number(emor5));
}

void PhotometricCalibrationWidget::setVignetteValues(double vC1, double vC2, double vC3) {
  ui->vignetteRenderWidget->setVignetteParams(vC1, vC2, vC3);
  ui->labelCoef1->setText(QString::number(vC1));
  ui->labelCoef2->setText(QString::number(vC2));
  ui->labelCoef3->setText(QString::number(vC3));
}

void PhotometricCalibrationWidget::onRenderVignetteClicked() {
  renderVignette = !renderVignette;
  ui->vignetteRenderWidget->setRenderPreview(renderVignette);
  if (renderVignette) {
    ui->buttonRenderVignette->setText(tr("Show curve"));
  } else {
    ui->buttonRenderVignette->setText((tr("Render vignette")));
  }
}
