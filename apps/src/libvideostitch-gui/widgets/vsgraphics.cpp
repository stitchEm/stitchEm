// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vsgraphics.hpp"

const QColor VSGraphicsScene::backgroundColor = QColor(18, 18, 18); /* FIXME: No hard coded colors please. */
const QBrush VSGraphicsScene::backgroundBrush = QBrush(VSGraphicsScene::backgroundColor, Qt::SolidPattern);

VSGraphicsScene::VSGraphicsScene(QObject *parent) : QGraphicsScene(parent) {}

void VSGraphicsScene::dragEnterEvent(QGraphicsSceneDragDropEvent *event) { event->ignore(); }

void VSGraphicsScene::dropEvent(QGraphicsSceneDragDropEvent *event) { event->ignore(); }

VSGraphicsView::VSGraphicsView(QWidget *parent) : QGraphicsView(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAttribute(Qt::WA_NoSystemBackground);
  viewport()->setAttribute(Qt::WA_OpaquePaintEvent);
  viewport()->setAttribute(Qt::WA_NoSystemBackground);
  setOptimizationFlags(QGraphicsView::IndirectPainting);
  setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
  setOptimizationFlags(QGraphicsView::DontClipPainter);
  setOptimizationFlags(QGraphicsView::DontSavePainterState);
  setOptimizationFlags(QGraphicsView::DontAdjustForAntialiasing);
  setCacheMode(QGraphicsView::CacheBackground);

  setStyleSheet("QGraphicsView { border-style: none; }");
}

void VSGraphicsView::showEvent(QShowEvent *event) { QWidget::showEvent(event); }

void VSGraphicsView::resizeEvent(QResizeEvent *event) {
  if (scene()) {
    fitInView(0, 0, scene()->width(), scene()->height(), Qt::KeepAspectRatio);
  }
  QWidget::resizeEvent(event);
}
