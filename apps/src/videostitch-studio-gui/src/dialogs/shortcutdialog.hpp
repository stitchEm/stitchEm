// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SHORTCUTDIALOG_HPP
#define SHORTCUTDIALOG_HPP

#include <QDialog>

namespace Ui {
class ShortcutDialog;
}

class ShortcutDialog : public QDialog {
  Q_OBJECT

 public:
  explicit ShortcutDialog(QWidget *parent = 0);
  ~ShortcutDialog();

 private:
  Ui::ShortcutDialog *ui;
};

#endif  // SHORTCUTDIALOG_HPP
