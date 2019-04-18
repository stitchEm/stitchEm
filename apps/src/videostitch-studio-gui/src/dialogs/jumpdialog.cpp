// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "jumpdialog.hpp"
#include "ui_jumpdialog.h"

JumpDialog::JumpDialog(const frameid_t firstFrame, const frameid_t lastFrame, QWidget* const parent)
    : QDialog(parent, Qt::WindowTitleHint), ui(new Ui::JumpDialog) {
  ui->setupUi(this);
  ui->frameSpinBox->setMinimum(firstFrame);
  ui->frameSpinBox->setMaximum(lastFrame);
  ui->frameSpinBox->setFocus();
  ui->frameSpinBox->selectAll();
  connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &JumpDialog::close);
  connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &JumpDialog::onButtonAcceptClicked);
}

JumpDialog::~JumpDialog() { delete ui; }

void JumpDialog::onButtonAcceptClicked() {
  emit reqSeek(ui->frameSpinBox->value());
  close();
}
