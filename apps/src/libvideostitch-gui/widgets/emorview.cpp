// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "emorview.hpp"

#include "libvideostitch/inputDef.hpp"

#include "libvideostitch/emor.hpp"

#include <QPainter>

static const float emorViewSize = 189;

EmorView::EmorView(QWidget *parent) : QWidget(parent), emorA(0.0), emorB(0.0), emorC(0.0), emorD(0.0), emorE(0.0) {}

void EmorView::paintEvent(QPaintEvent *) {
  QPainter painter(this);

  painter.setPen(QColor(95, 95, 95));

  painter.drawRect(0, 0, emorViewSize, emorViewSize);

  VideoStitch::Core::EmorResponseCurve emorCurve(emorA, emorB, emorC, emorD, emorE);

  const float *responseCurve = emorCurve.getResponseCurve();
  QPointF *qpoints = new QPointF[1024];

  for (int i = 0; i < 1024; ++i) {
    qpoints[i].setX(float(i) / 1024 * emorViewSize);
    qpoints[i].setY((1 - responseCurve[i]) * emorViewSize);
  }

  painter.setRenderHint(QPainter::Antialiasing);
  painter.drawPolyline(qpoints, 1024);
}
