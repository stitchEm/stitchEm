// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VSCURVESTREEWIDGET_HPP
#define VSCURVESTREEWIDGET_HPP

#include "timeline/curvegraphicsitem.hpp"
#include "timeline/timeline.hpp"

#include "libvideostitch-gui/caps/guistatecaps.hpp"

#include "libvideostitch/curves.hpp"

#include <QTreeWidget>
#include <vector>

class PostProdProjectDefinition;

/**
 * @brief The CurvesTreeWidget class is a class that displays the different curves that can be displayed in the
 * timeline. This class is used as a communication bridge between the timeline and the stitcher. If you need to update
 * the curves the timeline, or from the stitcher: this is the class you need to use.
 */
class CurvesTreeWidget : public QTreeWidget, public GUIStateCaps {
  Q_OBJECT

 public:
  explicit CurvesTreeWidget(QWidget *parent = nullptr);
  ~CurvesTreeWidget();

  /**
   * Sets the timeline to use.
   */
  void setTimeline(Timeline *t) {
    timeline = t;
    connect(this, SIGNAL(reqRemoveNamedCurve(TimelineItemPayload)), timeline,
            SLOT(removeNamedCurve(TimelineItemPayload)));
  }

  /**
   * Populates the curves tree from a stitcher.
   */
  void populate(const PostProdProjectDefinition &project);

  /**
   * Clears the tree.
   */
  void clear();
  /**
   * Removes all items in previouslySelectedItems from the timeline and clears previouslySelectedItems.
   */
  void removePreviouslySelectedFromTimeline();

 public slots:
  void onProjectOrientable(bool orientable);
  void updateCurve(VideoStitch::Core::Curve *curve, CurveGraphicsItem::Type type, int inputId);
  void updateQuaternionCurve(VideoStitch::Core::QuaternionCurve *curve, CurveGraphicsItem::Type type, int inputId);

 signals:
  /**
   * @brief Requests the State Manager to initiate a specific state transition.
   * @param s is the requested state.
   */
  void reqChangeState(GUIStateCaps::State s);
  void reqResetCurve(CurveGraphicsItem::Type type, int inputId = -1);
  void reqRemoveNamedCurve(const TimelineItemPayload &payload);
 private slots:
  /**
   * @brief Changes the widget's stats to the given state.
   * @param s State you want to switch to.
   */
  void changeState(GUIStateCaps::State s);

  /**
   * Updates the timeline contents. Triggered when the selection of the curves tree changes.
   */
  void updateTimelineContents();
  void onItemExpanded(QTreeWidgetItem *item);
  void onItemCollapsed(QTreeWidgetItem *item);
  void onItemClicked(QTreeWidgetItem *item, bool fromEntered = false);

 private:
  void mousePressEvent(QMouseEvent *event);
  void resetItem(QTreeWidgetItem *item);
  void setChildrenSelected(QTreeWidgetItem *item, bool selected);

  const static QColor GLOBAL_EV_COLOR;
  const static QColor CB_COLOR;
  const static QColor CR_COLOR;
  const static QColor STABILIZED_YAW_COLOR;
  const static QColor STABILIZED_PITCH_COLOR;
  const static QColor STABILIZED_ROLL_COLOR;
  void replaceGlobalCurve(VideoStitch::Core::Curve *curve, CurveGraphicsItem::Type type);
  void replaceGlobalQuaternionCurve(VideoStitch::Core::QuaternionCurve *curve, CurveGraphicsItem::Type type);
  void replaceInputCurve(VideoStitch::Core::Curve *curve, CurveGraphicsItem::Type type, int index);
  /**
   * Adds a child with a curve to a parent. If @a curve is null, the item has no associated curve.
   */
  void addCurveChild(QTreeWidgetItem *parent, const QString &name, const VideoStitch::Core::Curve *curve,
                     CurveGraphicsItem::Type curveType, double minValue, double maxValue, int inputId,
                     const QColor &color = QColor());

  Timeline *timeline;
  QSet<QTreeWidgetItem *> previouslySelectedItems;
  std::vector<VideoStitch::Core::Curve *> garbageToCollect;
  QList<QTreeWidgetItem *> userSelectedItems;
};

#endif  // VSCURVESTREEWIDGET_HPP
