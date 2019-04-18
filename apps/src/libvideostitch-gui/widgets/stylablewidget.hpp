// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QPainter>
#include <QPaintEvent>
#include <QStyle>
#include <QStyleOption>

/**
 * This macro shall be used by class derived from QWidget.
 * The paintEvent overload is required by Qt to handle stylesheet, see:
 * http://doc.qt.io/qt-5/stylesheet-reference.html
 */
#define Q_MAKE_STYLABLE                                        \
 protected:                                                    \
  virtual void paintEvent(QPaintEvent *) override {            \
    QStyleOption opt;                                          \
    opt.init(this);                                            \
    QPainter p(this);                                          \
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this); \
  }
