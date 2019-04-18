// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PROGRESSDIALOG_HPP
#define PROGRESSDIALOG_HPP

#include <QDialog>
class ProgressReporterWrapper;
namespace Ui {
class ModalProgressDialog;
}

class VS_GUI_EXPORT ModalProgressDialog : public QDialog {
  Q_OBJECT

 public:
  explicit ModalProgressDialog(const QString title, QWidget *parent);
  ~ModalProgressDialog();
  ProgressReporterWrapper *getReporter();

 public slots:
  void show();
  /*
   * Override the "reject" method (triggered by hitting the escape key on MacOS) to have it behave like "close"
   * otherwise it just hides the ProgressDialog without closing it
   */
  virtual void reject() override;

 private slots:
  void tryToCancel();

 private:
  virtual void closeEvent(QCloseEvent *event) override;
  virtual void showEvent(QShowEvent *event) override;

  Ui::ModalProgressDialog *ui;
};

#endif  // PROGRESSDIALOG_HPP
