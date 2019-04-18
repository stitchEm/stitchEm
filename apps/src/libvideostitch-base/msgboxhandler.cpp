// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "msgboxhandler.hpp"

#include "logmanager.hpp"

MsgBoxHandler::MsgBoxHandler(QObject *parent) : QObject(parent), Singleton<MsgBoxHandler>() {
  qRegisterMetaType<QFlags<QMessageBox::StandardButton>>("QFlags<QMessageBox::StandardButton>");
  connect(this, SIGNAL(reqGenericPrivate(QString, QString, QString, QFlags<QMessageBox::StandardButton>, QString)),
          this, SLOT(genericPrivate(QString, QString, QString, QFlags<QMessageBox::StandardButton>, QString)),
          Qt::QueuedConnection);
}

void MsgBoxHandler::generic(QString str, QString title, QString iconPath, QFlags<QMessageBox::StandardButton> buttons,
                            QString detailedText) {
  emit reqGenericPrivate(str, title, iconPath, buttons, detailedText);
}

int MsgBoxHandler::genericSync(QString str, QString title, QString iconPath,
                               QFlags<QMessageBox::StandardButton> buttons, QString detailedText) {
  return genericPrivate(str, title, iconPath, buttons, detailedText);
}

int MsgBoxHandler::genericPrivate(QString str, QString title, QString iconPath,
                                  QFlags<QMessageBox::StandardButton> buttons, QString detailedText) {
  QString level(QStringLiteral("Info"));
  if (iconPath == CRITICAL_ERROR_ICON) {
    level = QStringLiteral("Error");
  } else if (iconPath == WARNING_ICON) {
    level = QStringLiteral("Warning");
  }
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
      QString("[%0] Showing dialog: %1\n%2\n%3").arg(level).arg(title).arg(str).arg(detailedText));

  QMessageBox msgBox;
  msgBox.setWindowTitle(title);
  msgBox.setText(str);
  msgBox.setInformativeText(detailedText);
  msgBox.setWindowModality(Qt::ApplicationModal);
  msgBox.setStandardButtons(buttons);
  msgBox.setWindowFlags(Qt::WindowTitleHint);
  if (iconPath != NO_ICON) {
    QPixmap icon;
    icon.load(iconPath);
    msgBox.setIconPixmap(icon.scaled(24, 24, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  }
  msgBox.adjustSize();
  return msgBox.exec();
}
