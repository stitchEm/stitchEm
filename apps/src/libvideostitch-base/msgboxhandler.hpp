// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"
#include "singleton.hpp"

#include <QMessageBox>

#define WARNING_ICON ":/assets/icon/common/warning.png"
#define INFORMATION_ICON ":/assets/icon/common/info.png"
#define CRITICAL_ERROR_ICON ":/assets/icon/common/error.png"
#define QUESTION_ICON ":/assets/icon/common/question.png"
#define NO_ICON ""

/**
 * @brief The MsgBoxHandler class provides thread-safe generic message boxes.
 */
class VS_COMMON_EXPORT MsgBoxHandler : public QObject, public Singleton<MsgBoxHandler> {
  friend class Singleton<MsgBoxHandler>;
  Q_OBJECT

 public:
  /**
   * @brief generic requests a generic message box. Thread-safe.
   * @param str QString containing the message.
   * @param title Title name of the message box.
   * @param iconPath The icon for the message box. WARNING_ICON, INFORMATION_ICON, NO_ICON helpers can be used.
   * @param buttons [optional] Information on the buttons.
   * @param detailedText [optional] A detailed text if any.
   */
  void generic(QString str, QString title, QString iconPath,
               QFlags<QMessageBox::StandardButton> buttons = QFlags<QMessageBox::StandardButton>(QMessageBox::Ok),
               QString detailedText = "");

  /**
   * @brief generic requests a message box with a return value. Not thread-safe, must be executed from the GUI thread.
   * @param str QString containing the message.
   * @param title Title name of the message box.
   * @param iconPath The icon for the message box. WARNING_ICON, INFORMATION_ICON, NO_ICON helpers can be used.
   * @param buttons [optional] Information on the buttons.
   * @param detailedText [optional] A detailed text if any.
   * @return the QMessageBox return value.
   */
  int genericSync(QString str, QString title, QString iconPath,
                  QFlags<QMessageBox::StandardButton> buttons = QFlags<QMessageBox::StandardButton>(QMessageBox::Ok),
                  QString detailedText = "");

 signals:
  /**
   * @brief reqGeneric requests a generic message box. Thread-safe.
   */
  void reqGenericPrivate(QString str, QString title, QString iconPath, QFlags<QMessageBox::StandardButton> buttons,
                         QString detailedText);

 private slots:
  /**
   * @brief genericPrivate displays a generic message box. Must be called from the GUI thread via the reqGeneric signal.
   * @param str QString containing the message.
   * @param title Title name of the message box.
   * @param iconPath The icon for the message box. WARNING_ICON, INFORMATION_ICON, NO_ICON helpers can be used.
   * @param buttons [optional] Information on the buttons.
   * @param detailedText [optional] A detailed text if any.
   * @return the QMessageBox return value.
   */
  int genericPrivate(QString str, QString title, QString iconPath, QFlags<QMessageBox::StandardButton> buttons,
                     QString detailedText);

 private:
  explicit MsgBoxHandler(QObject *parent = nullptr);
};
