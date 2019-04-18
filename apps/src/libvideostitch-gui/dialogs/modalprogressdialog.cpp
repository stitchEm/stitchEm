// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "modalprogressdialog.hpp"
#include "ui_modalprogressdialog.h"

#include <QPushButton>

ModalProgressDialog::ModalProgressDialog(const QString title, QWidget *parent)
    : QDialog(parent->window(), Qt::WindowTitleHint), ui(new Ui::ModalProgressDialog) {
  ui->setupUi(this);
  setAttribute(Qt::WA_DeleteOnClose, false);
  setWindowTitle(title);
  connect(getReporter(), &ProgressReporterWrapper::reqProgressMessage, ui->label, &QLabel::setText);
  connect(ui->buttonBox->button(QDialogButtonBox::Cancel), &QPushButton::clicked, this,
          &ModalProgressDialog::tryToCancel);
  setModal(true);
  setWindowModality(Qt::WindowModal);
}

ModalProgressDialog::~ModalProgressDialog() { delete ui; }

void ModalProgressDialog::show() {
  QWidget *algoSender = static_cast<QWidget *>(sender());
  if (algoSender) {
    ui->label->setText(algoSender->windowTitle());
  }
  return QDialog::show();
}

void ModalProgressDialog::reject() { tryToCancel(); }

void ModalProgressDialog::tryToCancel() {
  ui->buttonBox->button(QDialogButtonBox::Cancel)->setEnabled(false);
  ui->buttonBox->button(QDialogButtonBox::Cancel)->setText(tr("Cancelling..."));
  ui->progressBar->cancel();
}

ProgressReporterWrapper *ModalProgressDialog::getReporter() { return ui->progressBar; }

void ModalProgressDialog::closeEvent(QCloseEvent *event) {
  QDialog::closeEvent(event);
  ui->progressBar->cancel();
}

void ModalProgressDialog::showEvent(QShowEvent *event) {
  ui->progressBar->reset();
  QDialog::showEvent(event);
}
