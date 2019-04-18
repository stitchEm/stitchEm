// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "constrainedgraphicsitemgroup.hpp"
#include "curvegraphicsitem.hpp"
#include "timelinescene.hpp"

#include "libvideostitch/curves.hpp"

#include <QGraphicsView>
#include <QPainterPath>

class ProjectDefinition;
class QScrollBar;
class QResizeEvent;

static const unsigned STABILIZATION_CURVES(3);
static const int MINIMAL_ELEMENTS(2);

/**
 * A class that represents a payload for a timeline item (curve, frame offset).
 */
class TimelineItemPayload {
 public:
  enum class Type { None, Curve };

  /**
   * An empty payload.
   */
  TimelineItemPayload()
      : type(Type::None),
        offset(-1),
        curve(nullptr),
        curveType(CurveGraphicsItem::InputExposure),
        minCurveValue(-180.0),
        maxCurveValue(180.0),
        inputId(-1) {}

  /**
   * A curve payload.
   */
  TimelineItemPayload(const QString& name, VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type curveType,
                      double minCurveValue, double maxCurveValue, int inputId, QColor color)
      : type(Type::Curve),
        offset(-1),
        name(name.toStdString()),
        curve(curve),
        curveType(curveType),
        minCurveValue(minCurveValue),
        maxCurveValue(maxCurveValue),
        inputId(inputId),
        color(color) {}

  bool operator<(const TimelineItemPayload& other) const {
    if (inputId == other.inputId && name != other.name) {
      return name < other.name;
    } else if (name == other.name && inputId == other.inputId) {
      return curveType < other.curveType;
    } else {
      return inputId < other.inputId;
    }
  }

  const Type type;
  // Only valid for type FrameOffset
  const int offset;
  // Only valid for type Curve
  const std::string name;
  VideoStitch::Core::Curve* const curve;  // Not owned.
  CurveGraphicsItem::Type curveType;
  const double minCurveValue;  // min curve value.
  const double maxCurveValue;  // max curve value.
  // For all types.
  const int inputId;  // -1 if not relevant (e.g. global curve)
  const QColor color;
};

static const double TIMELINE_HEIGHT = 360.0f;
static const double TIMELINE_WIDTH = 1000.0f;
static const double SCENE_HBORDER = 16.0f;
static const double SCENE_WIDTH = TIMELINE_WIDTH + 2 * SCENE_HBORDER;
static const double SCENE_HEIGHT = TIMELINE_HEIGHT;

Q_DECLARE_METATYPE(TimelineItemPayload);

/**
 * A Cursor graphics class.
 */
class TimelineCursor : public ConstrainedGraphicsItemGroup {
 public:
  /**
   * Creates a cursor.
   */
  TimelineCursor();
  virtual QRectF boundingRect() const override;
  virtual void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
  virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
  virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
};

/**
 * @brief The Timeline class is used to display a premiere/blender-like timeline.
 */
class Timeline : public QGraphicsView {
  Q_OBJECT
  Q_PROPERTY(QColor cursorLineColor READ getCursorLineColor WRITE setCursorLineColor DESIGNABLE true)
  Q_PROPERTY(QColor workingZoneColor READ getWorkingZoneColor WRITE setWorkingZoneColor DESIGNABLE true)
  Q_PROPERTY(QColor zeroLevelColor READ getZeroLevelColor WRITE setZeroLevelColor DESIGNABLE true)
  Q_PROPERTY(int selectedItemCount READ getSelectedItemCount)  // Custom property used in tst_timeline_selection
                                                               // (squish)
  Q_PROPERTY(bool stabilizationCurvesModified READ getStabilizationCurvesModified)  // Property used in
                                                                                    // tst_stabilization

 public:
  /**
   * The height of the scene, in pixels. Logical, different from the widget height.
   */
  explicit Timeline(QWidget* parent = 0);

  ~Timeline();
  void resetRange();
  void resetTimeline();
  void setProject(ProjectDefinition* p);

  /**
   * @brief visibleRect Computes the visible part of the scene.
   * @return  QRectF representing the visible part of the scene.
   */
  QRectF visibleRect() const;
  /**
   * Returns the minimum frame in the focus area.
   */
  qint64 minimum() const;
  /**
   * Returns the maximum frame in the focus area.
   */
  qint64 maximum() const;
  /**
   * Returns the current frame (always within the focus area).
   */
  qint64 value() const;
  /**
   * Add a curve. If a curve with this name already exists, does nothing and returns false.
   */
  bool addNamedCurve(const TimelineItemPayload& payload);
  void removeSelectedKeyframesAsync();

  QString getFramestringFromFrame(frameid_t frame) const;

  int getCursorPosition() const;
  int getLowerBoundPosition() const;
  int getUpperBoundPosition() const;

  /**
   *  QSS properties
   */
  void setCursorLineColor(QColor color);
  QColor getCursorLineColor() const;
  void setWorkingZoneColor(QColor color);
  QColor getWorkingZoneColor() const;
  QColor getZeroLevelColor() const;
  void setZeroLevelColor(QColor color);

  int getSelectedItemCount() const { return scene()->selectedItems().size(); }

  bool getStabilizationCurvesModified() const {
    bool modified = true;
    unsigned count = 0;
    for (auto curve : curves) {
      if (curve.second->getType() == CurveGraphicsItem::Type::Stabilization) {
        ++count;
        modified = modified && (curve.second->getPathItem()->path().elementCount() > MINIMAL_ELEMENTS);
      }
    }
    return modified && count == STABILIZATION_CURVES;
  }

  void clearTimeline();

  /**
   * @brief return the next keyframe
   * @return valid keyframe
   */
  int nextKeyFrame() const;

  /**
   * @brief return the previous keyframe
   * @return valid keyframe
   */
  int prevKeyFrame() const;

 signals:
  void reqSeek(frameid_t frame);
  void reqCurveChanged(SignalCompressionCaps* comp, VideoStitch::Core::Curve*, CurveGraphicsItem::Type type,
                       int inputId);
  void reqQuaternionCurveChanged(SignalCompressionCaps* comp, VideoStitch::Core::QuaternionCurve*,
                                 CurveGraphicsItem::Type type, int inputId);
  void reqResetCurves();
  void reqResetCurve(CurveGraphicsItem::Type type, int inputId = -1);
  void reqRemoveSelectedKeyframes();
  void reqRefreshTicks();
  void reqRefreshCurves();
  void reqUpdateZoomSliders(double maxZoomX);
  void reqUpdateKeyFrames(bool hasKeyFrames);

 public slots:
  /**
   * Remove a curve.
   */
  void removeNamedCurve(const TimelineItemPayload& payload);
  void removeSelectedKeyframes();

  void addKeyframeHere();
  /**
   * @brief mapToPosition Maps a frame into a position in the scene
   * @param frame frame you need to get the position
   * @return X order position
   */
  double mapToPosition(qint64 frame) const;
  /**
   * @brief mapFromPosition maps a given position in the timeline to a frame
   * @param position X order position in the timeline
   * @return corresponding frame
   */
  int mapFromPosition(double position) const;
  /**
   * @brief setValue Sets the timeline cursor to a given value.
   * @param frame Value to set the cursor to.
   */
  void setValue(frameid_t frame);
  /**
   * @brief moveCursorTo Sets the timeline cursor to a given value and emits "valueChanged"
   * @param frame Value to set the cursor to.
   */
  void moveCursorTo(frameid_t frame);
  /**
   * @brief setRange Sets the range of the timeline.
   * @param minimum New minimum of the timeline.
   * @param maximum New maximum of the timeline.
   */
  void setRange(frameid_t min, frameid_t max);
  void setBounds(qint64 min, qint64 max);
  void clearSelectedItems();
  void setZoomMultiplier(double mult);
  double getScaleValue() const;

 private slots:
  /**
   * @brief moveCursorTo Sets the timeline cursor to a given position and emits "valueChanged"
   * @param newPos New position of the cursor. The final position can be different of the given one if the cursor's
   * movements are constrained.
   */
  void moveCursorTo(const QPointF& newPos);

  /**
   * @brief on_curve_changed Slots called when a curve has changed
   * @param comp Signal compressor
   * @param curve Curve which has been updated
   * @param type type of the curve
   * @param inputId Input id, or -1 if global.
   */
  void on_curve_changed(SignalCompressionCaps* comp, VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type,
                        int inputId);
  void on_quaternion_curve_changed(SignalCompressionCaps* comp, VideoStitch::Core::QuaternionCurve* curve,
                                   CurveGraphicsItem::Type type, int inputId);

  /**
   * @brief sliderGrabbed Slot called when the slider has been grabbed.
   */
  void sliderGrabbed();
  /**
   * @brief sliderReleased Slot called when the slider has been released.
   */
  void sliderReleased();
  void setZoomLevel(int percentage);

 private:
  virtual void keyPressEvent(QKeyEvent* event) override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void resizeEvent(QResizeEvent* event) override;
  virtual void showEvent(QShowEvent* event) override;
  virtual void drawBackground(QPainter* painter, const QRectF& rect) override;
  virtual void paintEvent(QPaintEvent* event) override;
  void setAnchor(QGraphicsView::ViewportAnchor anchor);
  void fillViewport();
  void updateSliders();
  bool hasVisibleKeyFrames() const;
  void restoreScale();

  /**
   * Recomputes the scene rect so that the handle is visible.
   */
  void recomputeSceneRect();
  void moveCenterOfScrollbar(double proportion, QScrollBar* bar);
  double getMiddlePagestepPosition(QScrollBar* bar);

  ProjectDefinition* project;
  QGraphicsRectItem* timelineRect;  // Owned by the scene.
  TimelineCursor* cursor;           // Owned by the scene.
  typedef std::map<TimelineItemPayload, CurveGraphicsItem*> NamedItems;
  NamedItems curves;

  frameid_t minFrame;
  frameid_t maxFrame;
  qint64 lowerBound;
  qint64 upperBound;
  bool sliderHeld;

  /**
   * QSS properties
   */

  QColor cursorLineColor;
  QColor workingZoneColor;
  QColor zeroLevelColor;

  double baseOffset;
  QList<QGraphicsItem*> selectedItems;
  double zoomMultiplier;
  int zoomPercentage;
  TimelineScene* timelineScene;
  bool mousePressed;
  QPoint antiSelectionStart;
  QSet<QGraphicsItem*> selectedBeforeMove;
};
