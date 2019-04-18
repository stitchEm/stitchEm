// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QDialog>
#include "ui_aboutwindow.h"

class VS_GUI_EXPORT AboutWidget : public QWidget, public Ui::AboutWidgetClass {
  Q_OBJECT
 public:
  explicit AboutWidget(QString version, QWidget* const parent = nullptr);

  ~AboutWidget();

 private slots:
  void onButtonWebSiteClicked();
};
