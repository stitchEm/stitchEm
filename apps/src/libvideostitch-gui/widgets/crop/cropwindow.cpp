// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropwindow.hpp"
#include "videostitcher/projectdefinition.hpp"
#include "cropinputtab.hpp"

#include <QDesktopWidget>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QPushButton>

CropWindow::CropWindow(ProjectDefinition* p, InputLensClass::LensType t, const int extended, QWidget* parent)
    : QDialog(parent), cropWidget(p, t, extended, this) {
  QDialogButtonBox* buttonsBox = new QDialogButtonBox(this);
  QPushButton* buttonApply = new QPushButton(tr("Apply"), this);
  QPushButton* buttonCancel = new QPushButton(tr("Cancel"), this);
  buttonApply->setObjectName("buttonApply");
  buttonCancel->setObjectName("buttonCancel");
  buttonApply->setProperty("vs-button-medium", true);
  buttonCancel->setProperty("vs-button-medium", true);
  buttonsBox->addButton(buttonApply, QDialogButtonBox::AcceptRole);
  buttonsBox->addButton(buttonCancel, QDialogButtonBox::RejectRole);
  cropWidget.getHorizontalLayout()->addWidget(buttonsBox);
  connect(buttonsBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonsBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

  const int defaultMargin = 0;
  QVBoxLayout* layout = new QVBoxLayout(this);
  layout->setContentsMargins(defaultMargin, defaultMargin, defaultMargin, defaultMargin);
  layout->addWidget(&cropWidget);

  setModal(true);
  setWindowFlags((windowFlags() | Qt::CustomizeWindowHint | Qt::MSWindowsFixedSizeDialogHint) &
                 ~Qt::WindowCloseButtonHint & ~Qt::WindowContextHelpButtonHint);
}

CropWindow::~CropWindow() {}

CropWidget& CropWindow::getCropWidget() { return cropWidget; }

void CropWindow::reject() { QDialog::reject(); }

void CropWindow::accept() {
  cropWidget.applyCrop();
  QDialog::accept();
}
