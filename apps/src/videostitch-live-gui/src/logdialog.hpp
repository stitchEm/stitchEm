// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LOGDIALOG_HPP
#define LOGDIALOG_HPP

#include <QFrame>
#include <QSize>
#include <QScopedPointer>

class LogWidget;
class QVBoxLayout;
class QHBoxLayout;
class QLabel;
class QPushButton;
class LogDialog : public QFrame {
  Q_OBJECT
 public:
  explicit LogDialog(QWidget* const parent = nullptr);

  ~LogDialog();

  void updateSize(const QSize& value);

 private slots:
  void onLogLevelChanged(const int level);

 private:
  QScopedPointer<QFrame> background;

  QScopedPointer<QVBoxLayout> layoutBackground;

  QScopedPointer<QHBoxLayout> layoutHorizontal;

  QScopedPointer<QPushButton> buttonClose;

  QScopedPointer<QLabel> labelTitle;

  QScopedPointer<LogWidget> logWidget;
};

#endif  // LOGDIALOG_HPP
