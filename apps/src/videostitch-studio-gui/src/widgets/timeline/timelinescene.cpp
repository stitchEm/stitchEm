// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QGraphicsItem>
#include <QTransform>
#include "timeline.hpp"
#include "timelinescene.hpp"
#include "curvegraphicsitem.hpp"

#include <QGraphicsSceneMouseEvent>

TimelineScene::TimelineScene(QObject *parent) : QGraphicsScene(parent) {}

void TimelineScene::addToExclusionList(QGraphicsItem *item) { exclusionList.insert(item); }

QSet<QGraphicsItem *> TimelineScene::getExclusionList() const { return exclusionList; }

QSet<QGraphicsItem *> TimelineScene::getExcludedFromSelection() const { return toExcludeFromSelection; }

void TimelineScene::addToUnselectionSet(QSet<QGraphicsItem *> items) {
  toExcludeFromSelection = toExcludeFromSelection.unite(items);
}

void TimelineScene::addToUnselectionSet(QGraphicsItem *item) { toExcludeFromSelection << item; }

void TimelineScene::removeToUnselectionSet(QSet<QGraphicsItem *> items) {
  toExcludeFromSelection = toExcludeFromSelection.subtract(items);
}

void TimelineScene::clearExcludedFromSelectionList() { toExcludeFromSelection.clear(); }

void TimelineScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) { QGraphicsScene::mouseReleaseEvent(event); }

void TimelineScene::mousePressEvent(QGraphicsSceneMouseEvent *event) {
  QGraphicsItem *itemUnderMouse = itemAt(event->scenePos().x(), event->scenePos().y(), views()[0]->transform());

  if (!itemUnderMouse || exclusionList.contains(itemUnderMouse)) {
    QList<QGraphicsItem *> items = selectedItems();
    QGraphicsScene::mousePressEvent(event);
    foreach (QGraphicsItem *item, items) { item->setSelected(true); }
  } else {
    QGraphicsScene::mousePressEvent(event);
    if (event->modifiers() == Qt::AltModifier && itemUnderMouse && !exclusionList.contains(itemUnderMouse)) {
      toExcludeFromSelection << itemUnderMouse;
    }
  }
}
