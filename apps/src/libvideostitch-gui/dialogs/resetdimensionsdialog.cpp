// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "resetdimensionsdialog.hpp"
#include "ui_resetdimensionsdialog.h"

#include "mainwindow/msgboxhandlerhelper.hpp"

#include <QPushButton>

#define SIZE_SPINBOX_STEP 16

ResetDimensionsDialog::ResetDimensionsDialog(const QString ptvWhichFailed, unsigned int panoWidth,
                                             unsigned int panoHeight, QWidget *parent)
    : QDialog(parent, Qt::WindowTitleHint), ui(new Ui::ResetDimensionsDialog) {
  ui->setupUi(this);
  //: Error popup tile when out of GPU resource. %0 is the project name
  setWindowTitle(tr("The project %0 failed to initialize.").arg(ptvWhichFailed));
  QString message =
      tr("Your GPU doesn't have enough memory to stitch a %0x%1 image.\n"
         "Please reduce the panorama resolution in order to continue.\n"
         "You can also close the project, and create or load a new project.");
  ui->warningLabel->setText(message.arg(QString::number(panoWidth), QString::number(panoHeight)));
  ui->widthSpinBox->setMaximum(std::numeric_limits<short>::max());
  ui->heightSpinBox->setMaximum(std::numeric_limits<short>::max());
  ui->widthSpinBox->setSingleStep(SIZE_SPINBOX_STEP);
  ui->heightSpinBox->setSingleStep(SIZE_SPINBOX_STEP);
  ui->widthSpinBox->setKeyboardTracking(false);
  ui->heightSpinBox->setKeyboardTracking(false);
  ui->buttonBox->button(QDialogButtonBox::Ok)->setText(tr("Set new size"));
  ui->buttonBox->button(QDialogButtonBox::Close)->setText(tr("Close project"));
  const QImage warning = QImage(WARNING_ICON);
  ui->iconLabel->setPixmap(QPixmap::fromImage(warning).scaled(32, 32, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  adjustSize();
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

  // calculate new proper dimensions
  unsigned int newHeight = panoHeight / 2;
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromHeight(newHeight);
  updateSizeSpinBoxes(size);

  connect(ui->widthSpinBox, &QSpinBox::editingFinished, this, &ResetDimensionsDialog::onWidthChanged,
          Qt::UniqueConnection);
  connect(ui->heightSpinBox, &QSpinBox::editingFinished, this, &ResetDimensionsDialog::onHeightChanged,
          Qt::UniqueConnection);
}

ResetDimensionsDialog::~ResetDimensionsDialog() {}

unsigned int ResetDimensionsDialog::getNewPanoWidth() const { return ui->widthSpinBox->value(); }

unsigned int ResetDimensionsDialog::getNewPanoHeight() const { return ui->heightSpinBox->value(); }

void ResetDimensionsDialog::updateSizeSpinBoxes(VideoStitch::Util::PanoSize size) {
  if (ui->widthSpinBox->value() != size.width) {
    ui->widthSpinBox->setValue(size.width);
  }
  if (ui->heightSpinBox->value() != size.height) {
    ui->heightSpinBox->setValue(size.height);
  }
}

void ResetDimensionsDialog::onWidthChanged() {
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromWidth(ui->widthSpinBox->value());
  updateSizeSpinBoxes(size);
}

void ResetDimensionsDialog::onHeightChanged() {
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromHeight(ui->heightSpinBox->value());
  updateSizeSpinBoxes(size);
}
