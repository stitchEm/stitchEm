// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <QMainWindow>
#include <QStateMachine>
#include "tasktable.hpp"

class Packet;
class QDropEvent;

namespace Ui {
class BatchWindow;
}

class BatchWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit BatchWindow(const QString &fileToOpen = QString(), QWidget *parent = nullptr);
  ~BatchWindow();
 signals:
  void signalDeviceSelectionChanged(QList<int> devices);
 private slots:
  void openVS(QString fileToOpen);
  void vsProcessTerminated();
  /**
   *  @brief Processes the messages sent by the application.
   */
  void processMessage(const Packet &packet);
  void onDeviceSelectionChanged();
  void on_actionOpen_Project_triggered();

  void on_actionRemove_Selected_triggered();
  void on_stitchButton_clicked();
  void onRemoveAllButtonClicked();
  void updateButtonStates();

 private:
  QList<int> getSelectedDevices();

  void dropEvent(QDropEvent *e);
  void dragMoveEvent(QDragMoveEvent *);
  void dragEnterEvent(QDragEnterEvent *);
  void closeEvent(QCloseEvent *event);

  Ui::BatchWindow *ui;
  QStateMachine stateMachine;
  QState *idle;
  QState *processing;
  QProcess *vsProcess;
};

#endif  // MAINWINDOW_HPP
