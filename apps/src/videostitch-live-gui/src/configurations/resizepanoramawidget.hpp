// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef RESIZEPANORAMAWIDGET_HPP
#define RESIZEPANORAMAWIDGET_HPP

#include <QWidget>
#include "ui_resizepanoramawidget.h"

class ResizePanoramaWidget : public QWidget, public Ui::ResizePanoramaWidgetClass {
  Q_OBJECT
 public:
  explicit ResizePanoramaWidget(const int width, const int height, QWidget* const parent = nullptr);

  ~ResizePanoramaWidget();
};

#endif  // RESIZEPANORAMAWIDGET_HPP
