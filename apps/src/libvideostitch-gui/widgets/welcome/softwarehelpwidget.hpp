// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include <QFrame>

namespace Ui {
class SoftwareHelpWidget;
}

class SoftwareHelpWidget : public QFrame {
  Q_OBJECT

 public:
  explicit SoftwareHelpWidget(QWidget *parent = nullptr);
  ~SoftwareHelpWidget();

 private:
  void addHelpItem(const QString title, const QString url, const QString name);
  QScopedPointer<Ui::SoftwareHelpWidget> ui;
};
