// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "constrainedgraphicsitemgroup.hpp"

ConstrainedGraphicsItemGroup::ConstrainedGraphicsItemGroup(QGraphicsItem *parent)
    : QObject(), QGraphicsItemGroup(parent) {}

void ConstrainedGraphicsItemGroup::setConstraint(const QRectF &constraint) { constraintBounds = constraint; }

QRectF ConstrainedGraphicsItemGroup::getConstraintBounds() const { return constraintBounds; }

QVariant ConstrainedGraphicsItemGroup::itemChange(GraphicsItemChange change, const QVariant &value) {
  if (change == ItemPositionChange && scene()) {
    // value is the new position.
    QPointF newPos = value.toPointF();
    if (!constraintBounds.contains(newPos.toPoint())) {
      // Keep the item inside the scene rect.
      newPos.setX(qMin(constraintBounds.right(), qMax(newPos.x(), constraintBounds.left())));
      newPos.setY(qMin(constraintBounds.bottom(), qMax(newPos.y(), constraintBounds.top())));
      emit newPosition(newPos.toPoint());
      return newPos;
    }
  }
  return QGraphicsItem::itemChange(change, value);
}
