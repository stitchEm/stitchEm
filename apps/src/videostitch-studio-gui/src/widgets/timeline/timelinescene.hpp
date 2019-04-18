// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QGraphicsScene>
#include <QSet>

class QGraphicsSceneMouseEvent;

/**
 * @brief The TimelineScene class is a class which overloads several QGraphicsScene methods to allow
 *        specific features for selection/unselection.
 *        Remove+Click/Drag allows to add items to the selection
 *        Normal Click/Drag allows to define a basic selection
 *        Shift+Click/Drag allows to add items to the selection
 *
 *        This class contains an exclusion set, that exclusion set lists the items that have to be ignored.
 *        It also contains a "toExcludeFromSelection" set. These items are the items that have to be removed from the
 * selection. Finally, there is the "beforeSelection" set. It contains the "already" selected items that will have to be
 * added back once the click/drag event will be finished.
 */
class TimelineScene : public QGraphicsScene {
  Q_OBJECT
 public:
  explicit TimelineScene(QObject *parent = nullptr);
  /**
   * @brief Adds an item to the exclusion list.
   * @param item Item to add to the selection
   */
  void addToExclusionList(QGraphicsItem *item);
  QSet<QGraphicsItem *> getExclusionList() const;
  QSet<QGraphicsItem *> getExcludedFromSelection() const;
  /**
   * @brief Adds a set items to the excludeFromSelection list.
   * @param items Set of items that will be added to the unselection set.
   */
  void addToUnselectionSet(QSet<QGraphicsItem *> items);
  /**
   * @brief Adds an item to the excludeFromSelection list.
   * @param item Item that will be added to the unselection set.
   */
  void addToUnselectionSet(QGraphicsItem *item);
  /**
   * @brief Removes a set of items from the unselection set.
   * @param items
   */
  void removeToUnselectionSet(QSet<QGraphicsItem *> items);
  /**
   * @brief Clears the unselections et.
   */
  void clearExcludedFromSelectionList();

 private:
  virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
  virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
  QSet<QGraphicsItem *> exclusionList;
  QSet<QGraphicsItem *> toExcludeFromSelection;
  QSet<QGraphicsItem *> beforeSelection;
};
