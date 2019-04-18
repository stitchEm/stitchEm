// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef WELCOMEDIALOG_HPP
#define WELCOMEDIALOG_HPP

#include <QDialog>

namespace Ui {
class WelcomeDialog;
}

class WelcomeDialog : public QDialog {
  Q_OBJECT

 public:
  explicit WelcomeDialog(QWidget* const parent = nullptr);
  ~WelcomeDialog();

 private:
  Ui::WelcomeDialog* ui;
};

#endif  // WELCOMEDIALOG_HPP
