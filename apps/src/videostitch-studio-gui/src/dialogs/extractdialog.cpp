// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QFileDialog>
#include "extractdialog.hpp"
#include "ui_extractdialog.h"
#include "../mainwindow/mainwindow.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"

/**
 * Static variables/accessors that need to be shared accross the instances.
 */
QString ExtractDialog::lastCalibrationDirectory = QString();

void ExtractDialog::setLastCalibrationDirectory(const QString &directory) { lastCalibrationDirectory = directory; }

QString ExtractDialog::getLastCalibrationDirectory() { return lastCalibrationDirectory; }

void ExtractDialog::resetLastCalibrationDirectory() { lastCalibrationDirectory = QString(); }

ExtractDialog::ExtractDialog(QWidget *parent, QString &dir)
    : QDialog(parent, Qt::WindowTitleHint), ui(new Ui::ExtractDialog), m_dir(dir) {
  ui->setupUi(this);
  setWindowTitle(tr("Extract inputs' frames"));
  if (lastCalibrationDirectory.isEmpty() && lastCalibrationDirectory.isNull()) {
    m_dir = QDir::current().path();
  } else {
    m_dir = getLastCalibrationDirectory();
  }
  ui->labelPath->setText(m_dir);
  adjustSize();
}

ExtractDialog::~ExtractDialog() { delete ui; }

void ExtractDialog::on_browseButton_clicked() {
  QString selectedDir = QFileDialog::getExistingDirectory(this, tr("Select folder"), QDir::currentPath());

  while ((!QFileInfo(selectedDir).isWritable() && !(selectedDir.isNull() || selectedDir.isEmpty()))) {
    if (MsgBoxHandler::getInstance()->genericSync(
            tr("You do not have the right to write in %0. Do you want to select another directory?").arg(selectedDir),
            tr("Wrong permissions"), WARNING_ICON, QMessageBox::Retry | QMessageBox::No) == QMessageBox::No) {
      return;
    }
    selectedDir = QFileDialog::getExistingDirectory(this, tr("Select folder"), QDir::currentPath());
  }

  if (selectedDir.isNull() || selectedDir.isEmpty()) {
    return;
  }
  m_dir = selectedDir;
  ui->labelPath->setText(m_dir);
}

void ExtractDialog::on_buttonBox_accepted() { setLastCalibrationDirectory(m_dir); }
