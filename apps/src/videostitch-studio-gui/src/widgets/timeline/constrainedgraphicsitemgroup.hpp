// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QGraphicsItemGroup>

/**
 * @brief The ConstrainedGraphicsItemGroup class is a class used to represent a group of a QGraphicsItems
 * Which movements are constrained within a QRect.
 */
class ConstrainedGraphicsItemGroup : public QObject, public QGraphicsItemGroup {
  Q_OBJECT
 public:
  explicit ConstrainedGraphicsItemGroup(QGraphicsItem *parent = 0);

  /**
   * @brief setConstraint Sets the Rect to constraint the item into.
   * @param constraint Constraint rectangle.
   */
  void setConstraint(const QRectF &constraint);

  QRectF getConstraintBounds() const;

 signals:
  /**
   * @brief newPosition Signal emitted when the item group's position has changed.
   * @param pos New position.
   */
  void newPosition(QPointF pos);

 private:
  virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

  QRectF constraintBounds;
};
