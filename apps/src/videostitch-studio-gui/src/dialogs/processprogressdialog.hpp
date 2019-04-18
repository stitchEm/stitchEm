// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QDialog>
#include <QProcess>
#include "libvideostitch-gui/mainwindow/wintaskbarprogress.hpp"

namespace Ui {
class ProcessProgressDialog;
}

class VSCommandProcess;

class ProcessProgressDialog : public QDialog {
  Q_OBJECT
 public:
  /**
   * @brief Constructor oif the progress dialog
   * @param args Arguments of videostitch-cmd
   * @param firstFrame First processed frame
   * @param lastFrame Last process frame
   * @param parent Parent of this widget (needed for heap allocation)
   */
  explicit ProcessProgressDialog(QStringList args, qint64 firstFrame, qint64 lastFrame, QWidget *parent = nullptr);
  ~ProcessProgressDialog();

 signals:
#ifdef Q_OS_WIN
  /**
   * @brief Interface for the thumbnail progression on Window 7/8
   */
  void reqChangeProgressValue(quint64 current, quint64 total);
  void reqChangeProgressState(TBPFLAG state);
#endif
 public slots:
  /**
   * @brief Exec function, inherited from QDialog and used  to return the values according to the process result
   */
  virtual int exec() override;
  /**
   * @brief Slot called when the process has finished
   * @param exitCode Exit code of the process
   * @param exitStatus Exit status of the process
   */
  void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
  /**
   * @brief Slot called when the user clicks on "show details..."
   * @param show true = show the details / false = hide
   */
  void showDetailsClicked(bool show);
  void setProgress(int current);

 protected:
  virtual void closeEvent(QCloseEvent *event) override;

 private:
  void showWarning();
  Ui::ProcessProgressDialog *ui;
  VSCommandProcess *commandProcess;
  QStringList commandArgs;
  int firstFrame;
  int lastFrame;
};
