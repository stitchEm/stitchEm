// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "curvegraphicsitem.hpp"

#include "timeline.hpp"
#include "timelinescene.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"

#include "libvideostitch/curves.hpp"

#include <QCursor>
#include <QGraphicsEllipseItem>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsView>
#include <QMenu>
#include <QPainter>
#include <QPen>
#include <QTransform>
#include <QVector2D>
#include <QStyleOptionGraphicsItem>
#include <QStyle>

#include <math.h>
#include <assert.h>

static const double GAMMA(6.0);
static const double EXPOSURE_GAMMA(1.5);

const QBrush CurveGraphicsItem::unselectedBrush = QBrush(QColor(255, 255, 255));
const QBrush CurveGraphicsItem::selectedBrush = QBrush(QColor(255, 0, 0));
const QBrush CurveGraphicsItem::nonselectableBrush = QBrush(QColor(128, 128, 128));

double CurveGraphicsItem::toSceneV(double v) const {
  double result;
  double value = std::max(std::min(v, maxValue), minValue);
  if (type == RedCorrection || type == BlueCorrection || type == GlobalBlueCorrection || type == GlobalRedCorrection) {
    value = toLogScale(value);
    result = atan(GAMMA * value) * (toLogScale((double)maxValue) - toLogScale((double)minValue)) / M_PI;
    result = (SCENE_HEIGHT) * (result - toLogScale((double)minValue)) /
             (toLogScale((double)maxValue) - toLogScale((double)minValue));
    result = SCENE_HEIGHT - result;

    result = std::min(std::max(result, 0.0), (double)SCENE_HEIGHT);
  } else if (type == GlobalOrientation) {
    return SCENE_HEIGHT / 2.0;
  } else {
    result = value;
    if (type == GlobalExposure || type == InputExposure) {
      result = atan(EXPOSURE_GAMMA * value) * ((double)maxValue - (double)minValue) / M_PI;
    }
    result = (SCENE_HEIGHT) * (result - (double)minValue) / (double)(maxValue - minValue);
    result = SCENE_HEIGHT - result;
  }
  return result;
}

double CurveGraphicsItem::fromSceneV(double v) const {
  double result;
  double value = std::max(std::min(v, (double)SCENE_HEIGHT), 0.0);
  if (type == RedCorrection || type == BlueCorrection || type == GlobalBlueCorrection || type == GlobalRedCorrection) {
    if (value == SCENE_HEIGHT) {
      return minValue;
    } else if (type == GlobalOrientation) {
      return (maxValue - minValue) / 2.0;
    } else if (value == 0) {
      return maxValue;
    } else {
      result = (double)(toLogScale((double)maxValue) - toLogScale((double)minValue)) * (value) / (SCENE_HEIGHT);
      result = (double)(toLogScale((double)maxValue)) - result;
      result = result * M_PI / (toLogScale((double)maxValue) - toLogScale((double)minValue));
      result = tan(result);
      result = fromLogScale(result / GAMMA);
    }

  } else {
    result = (double)(maxValue - minValue) * (value) / (SCENE_HEIGHT);
    result = (double)(maxValue)-result;
    if (type == GlobalExposure || type == InputExposure) {
      result = result * M_PI / ((double)maxValue - (double)minValue);
      result = tan(result);
      result = result / EXPOSURE_GAMMA;
    }
  }

  return result;
}

CurveGraphicsItem::Type CurveGraphicsItem::getType() const { return type; }

int CurveGraphicsItem::getInputId() const { return inputId; }

bool CurveGraphicsItem::hasKeyframe() const { return !handles.empty(); }

/**
 * A handle for editing curves.
 */
class CurveGraphicsItem::HandleBase : public QAbstractGraphicsShapeItem {
 public:
  /**
   * Creates a handle. If prevPoint (resp. nextPoint) is not NULL, the handle movement is limited by it.
   * @a strict makes the limitation strict.
   */
  HandleBase(CurveGraphicsItem* parent, Timeline* timeline, VideoStitch::Core::Point& point,
             VideoStitch::Core::Spline* spline, const VideoStitch::Core::Point* prevPoint,
             const VideoStitch::Core::Point* nextPoint, bool strict)
      : QAbstractGraphicsShapeItem(parent),
        spline(spline),
        prevPoint(prevPoint),
        nextPoint(nextPoint),
        view(timeline),
        point(point),
        strict(strict) {
    setPen(QPen(QColor(0, 0, 0)));
    setBrush(unselectedBrush);
    setPos(timeline->mapToPosition(point.t), parent->toSceneV(point.v));
    setFlag(QGraphicsItem::ItemIgnoresTransformations);
    if (parent->getType() != CurveGraphicsItem::GlobalOrientation) {
      setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemSendsGeometryChanges |
               QGraphicsItem::ItemIsSelectable);
      setCursor(QCursor(Qt::SizeAllCursor));
    } else {
      setFlags(QGraphicsItem::ItemSendsGeometryChanges | QGraphicsItem::ItemIsSelectable);
    }
  }

  virtual ~HandleBase() {}

  virtual QRectF boundingRect() const {
    // Make sure the handle always has the same size.
    QTransform transform = static_cast<const CurveGraphicsItem*>(parentItem())->view()->transform();
    const double scaleX = transform.inverted().map(QPointF(1.0f, 0.0f)).x();
    const double scaleY = transform.inverted().map(QPointF(0.0f, 1.0f)).y();
    return QRectF(-scaleX * HANDLE_SIZE / 2.0, -scaleY * HANDLE_SIZE / 2.0, scaleX * HANDLE_SIZE, scaleY * HANDLE_SIZE);
  }

  void mousePressEvent(QGraphicsSceneMouseEvent* event) {
    QAbstractGraphicsShapeItem::mousePressEvent(event);
    if (event->button() == Qt::LeftButton) {
      if (event->modifiers() == Qt::AltModifier) {
        dynamic_cast<TimelineScene*>(scene())->addToUnselectionSet(this);
        setSelected(false);
      }
    }
  }

  void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    bool previousSelection = isSelected();
    QAbstractGraphicsShapeItem::mouseReleaseEvent(
        event);  // Selects the item. Only allowing mousePressEvent to select it.
    if (previousSelection != isSelected()) {
      setSelected(previousSelection);
    }
    event->accept();
  }

  void mouseMoveEvent(QGraphicsSceneMouseEvent* event) { QAbstractGraphicsShapeItem::mouseMoveEvent(event); }

  QVariant itemChange(GraphicsItemChange change, const QVariant& value) {
    if (change == ItemPositionChange) {
      QPointF newPos = value.toPointF();
      if (newPos.y() <= 0) {
        newPos.setY(0);
      }
      if (newPos.y() >= SCENE_HEIGHT && !(pos().y() < SCENE_HEIGHT)) {
        newPos.setY(SCENE_HEIGHT);
      }

      if (newPos.x() >= SCENE_WIDTH - SCENE_HBORDER) {
        newPos.setX(SCENE_WIDTH - SCENE_HBORDER);
      }

      if (prevPoint && strict && view->mapFromPosition(newPos.x()) <= prevPoint->t) {
        newPos.setX(view->mapToPosition(prevPoint->t + 1));
        point.t = prevPoint->t + 1;
      } else if (prevPoint && view->mapFromPosition(newPos.x()) < prevPoint->t) {
        newPos.setX(view->mapToPosition(prevPoint->t));
        point.t = prevPoint->t;
      } else if (nextPoint && strict && view->mapFromPosition(newPos.x()) >= nextPoint->t) {
        newPos.setX(view->mapToPosition(nextPoint->t - 1));
        point.t = nextPoint->t - 1;
      } else if (nextPoint && view->mapFromPosition(newPos.x()) > nextPoint->t) {
        newPos.setX(view->mapToPosition(nextPoint->t));
        point.t = nextPoint->t;
      } else {
        // Snap.
        point.t = static_cast<int>(view->mapFromPosition(newPos.x()));
        newPos.setX(view->mapToPosition(point.t));
      }
      if (point.t < 0) {
        point.t = static_cast<int>(0);
        newPos.setX(view->mapToPosition(point.t));
      } else if (view->mapToPosition(point.t) > scene()->width()) {
        point.t = static_cast<int>(view->mapFromPosition(scene()->width()));
        newPos.setX(view->mapToPosition(point.t));
      }
      CurveGraphicsItem* parent = static_cast<CurveGraphicsItem*>(parentItem());
      point.v = parent->fromSceneV(newPos.y());

      parent->signalChanged();
      setToolTip("t: " + view->getFramestringFromFrame(point.t) + ", value: " + QString::number(point.v));
      return newPos;
    } else {
      return QAbstractGraphicsShapeItem::itemChange(change, value);
    }
  }

  const VideoStitch::Core::Point& getPoint() const { return point; }

 protected:
  // Not owned.
  VideoStitch::Core::Spline* const spline;
  const VideoStitch::Core::Point* const prevPoint;
  const VideoStitch::Core::Point* const nextPoint;
  Timeline* view;
  VideoStitch::Core::Point& point;

 private:
  const bool strict;
};
/**
 * Regular handle.
 */
class CurveGraphicsItem::Handle : public CurveGraphicsItem::HandleBase {
 public:
  Handle(CurveGraphicsItem* parent, Timeline* view, VideoStitch::Core::Spline* spline)
      : HandleBase(parent, view, spline->end, spline, spline->prev ? &spline->prev->end : NULL,
                   spline->next ? &spline->next->end : NULL, true) {
    setAcceptHoverEvents(true);
  }

 protected:
  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
    Q_UNUSED(option)
    Q_UNUSED(widget)
    if (option->state & QStyle::State_Selected) {
      setBrush(selectedBrush);
    } else {
      setBrush(unselectedBrush);
    }

    painter->setBrush(brush());
    QPen itemPen(pen());
    itemPen.setCosmetic(true);
    painter->setPen(itemPen);
    painter->drawRect(boundingRect());
  }

  virtual void refresh() {
    view->clearSelectedItems();
    static_cast<CurveGraphicsItem*>(parentItem())->signalChanged();
    static_cast<CurveGraphicsItem*>(parentItem())->rebuildHandles();
  }

  virtual void mousePressEvent(QGraphicsSceneMouseEvent* event) {
    HandleBase::mousePressEvent(event);
    if (event->button() == Qt::RightButton) {
      event->accept();
      QMenu menu;
      QAction* remove =
          (pos().x() == 0 || scene()->selectedItems().size()) ? NULL : menu.addAction("Remove this keyframe");
      QAction* removeSelected = menu.addAction("Remove selected keyframes");
      QAction* leftSmooth = NULL;
      if (spline->getType() == VideoStitch::Core::Spline::LineType && scene()->selectedItems().size() <= 1) {
        leftSmooth = menu.addAction("Make left smooth");
      }
      QAction* rightSmooth = NULL;
      if (spline->next && spline->next->getType() == VideoStitch::Core::Spline::LineType &&
          scene()->selectedItems().size() <= 1) {
        rightSmooth = menu.addAction("Make right smooth");
      }
      QAction* selectedAction = menu.exec(event->screenPos());
      if (selectedAction == NULL) {
        return;
      }
      if (selectedAction == remove && pos().x() != 0) {
        static_cast<CurveGraphicsItem*>(parentItem())->removeKeyframe(getPoint().t);
        // Do not touch *this after the call above, it does not exist anymore.
        return;
      } else if (leftSmooth && selectedAction == leftSmooth) {
        spline->makeCubic();
        refresh();
        // Do not touch *this after the call above, it does not exist anymore.
      } else if (rightSmooth && selectedAction == rightSmooth) {
        spline->next->makeCubic();
        refresh();
        // Do not touch *this after the call above, it does not exist anymore.
      } else if (selectedAction == removeSelected) {
        view->removeSelectedKeyframesAsync();
        return;
      }
    }
  }
};

/**
 * A handle for editing constant curves (with no keyframes).
 */
class CurveGraphicsItem::ConstantCurveHandle : public QAbstractGraphicsShapeItem {
 public:
  /**
   * Creates a handle. If prevPoint (resp. nextPoint) is not NULL, the handle movement is limited by it.
   * @a strict makes the limitation strict.
   */
  ConstantCurveHandle(CurveGraphicsItem* parent, Timeline* timeline, VideoStitch::Core::Curve* curve)
      : QAbstractGraphicsShapeItem(parent), curve(curve) {
    setPen(QPen(QColor(0, 0, 0)));
    setBrush(nonselectableBrush);
    setPos(timeline->mapToPosition(0), parent->toSceneV(curve->at(0)));
    setFlag(QGraphicsItem::ItemIgnoresTransformations);
    if (parent->getType() != CurveGraphicsItem::GlobalOrientation) {
      setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemSendsGeometryChanges |
               QGraphicsItem::ItemIsSelectable);
      setCursor(QCursor(Qt::SizeVerCursor));
    } else {
      setFlags(QGraphicsItem::ItemSendsGeometryChanges | QGraphicsItem::ItemIsSelectable);
    }
    // setAcceptsHoverEvents(true);
  }

  virtual ~ConstantCurveHandle() {}

  virtual QRectF boundingRect() const {
    // Make sure the handle always has the same size.
    QTransform transform = static_cast<const CurveGraphicsItem*>(parentItem())->view()->transform();
    const double scaleX = transform.inverted().map(QPointF(1.0f, 0.0f)).x();
    const double scaleY = transform.inverted().map(QPointF(0.0f, 1.0f)).y();
    return QRectF(-3.0 * scaleX * HANDLE_SIZE / 2.0, -scaleY * HANDLE_SIZE / 2.0, 3.0 * scaleX * HANDLE_SIZE,
                  scaleY * HANDLE_SIZE);
  }

  void mouseMoveEvent(QGraphicsSceneMouseEvent* event) { QAbstractGraphicsShapeItem::mouseMoveEvent(event); }

  QVariant itemChange(GraphicsItemChange change, const QVariant& value) {
    if (change == ItemPositionChange) {
      QPointF newPos = value.toPointF();
      // Keep within bounds.
      if (newPos.y() <= 0) {
        newPos.setY(0);
      }
      if (newPos.y() >= SCENE_HEIGHT && !(pos().y() < SCENE_HEIGHT)) {
        newPos.setY(SCENE_HEIGHT);
      }
      // Only move vertically.
      newPos.setX(pos().x());

      CurveGraphicsItem* parent = static_cast<CurveGraphicsItem*>(parentItem());
      curve->setConstantValue(parent->fromSceneV(newPos.y()));

      parent->signalChanged();
      setToolTip("constnt curve, value: " + QString::number(curve->at(0)));
      return newPos;
    } else {
      return QAbstractGraphicsShapeItem::itemChange(change, value);
    }
  }

  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setBrush(brush());
    QPen itemPen(pen());
    itemPen.setCosmetic(true);
    painter->setPen(itemPen);
    painter->drawRect(boundingRect());
  }

  virtual void refresh() {
    static_cast<CurveGraphicsItem*>(parentItem())->signalChanged();
    static_cast<CurveGraphicsItem*>(parentItem())->rebuildHandles();
  }

 private:
  // Not owned.
  VideoStitch::Core::Curve* const curve;
};

CurveGraphicsItem::CurveGraphicsItem(Timeline* view, VideoStitch::Core::Curve& curve, Type type, double minValue,
                                     double maxValue, int inputId, const QColor& color)
    : QObject(),
      QGraphicsItem(),
      timeline(view),
      curve(curve),
      type(type),
      pathItem(this),
      constantHandle(nullptr),
      parentView(view),
      comp(nullptr),
      minValue(minValue),
      maxValue(maxValue),
      inputId(inputId) {
  view->scene()->addItem(this);
  setPos(0.0f, 0.0f);
  QPen pen(color);
  pen.setCosmetic(true);
  pathItem.setPen(pen);
  rebuildHandles();
  rebuildPath();
  // Enable signals after init:
  comp = SignalCompressionCaps::createOwned();
}

CurveGraphicsItem::~CurveGraphicsItem() {
  for (HandleBaseVector::iterator it = handles.begin(); it != handles.end(); ++it) {
    delete *it;
  }
  delete constantHandle;
}

QRectF CurveGraphicsItem::boundingRect() const { return pathItem.boundingRect(); }

void CurveGraphicsItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(painter)
  Q_UNUSED(option)
  Q_UNUSED(widget)
  if (type == CurveGraphicsItem::Stabilization) {
    painter->setRenderHint(QPainter::Antialiasing, true);
  }
}

void CurveGraphicsItem::addKeyframe(int frame) {
  if (type == CurveGraphicsItem::GlobalOrientation) {
    return;
  }
  curve.splitAt(frame);
  rebuildHandles();
  signalChanged();
}

void CurveGraphicsItem::removeKeyframe(int frame) {
  curve.mergeAt(frame);
  rebuildHandles();
  signalChanged();
}

void CurveGraphicsItem::addToRemovalList(int frame) { removalList.push_back(frame); }

void CurveGraphicsItem::processRemovalList() {
  if (removalList.isEmpty()) {
    return;
  }

  qSort(removalList);
  for (int k = 0; k < (removalList.size() / 2); k++) {
    removalList.swap(k, removalList.size() - (1 + k));
  }

  for (int i = 0; i < removalList.size(); ++i) {
    curve.mergeAt(removalList[i]);
  }

  removalList.clear();
  rebuildHandles();
  signalChanged();
}

int CurveGraphicsItem::getPrevKeyFrame(int currentFrame) {
  int returnValue(currentFrame);
  bool found(hasKeyframe() && currentFrame <= getFrameFromIterator(handles.begin()));
  if (found) returnValue = getFrameFromIterator(handles.rbegin());

  for (HandleBaseVectorRevIterator it(handles.rbegin()); it != handles.rend() && !found; ++it) {
    returnValue = getFrameFromIterator<HandleBaseVectorRevIterator>(it);
    found = currentFrame > returnValue;
  }
  return returnValue;
}

int CurveGraphicsItem::getNextKeyFrame(int currentFrame) {
  int returnValue(currentFrame);
  bool found(hasKeyframe() && currentFrame >= getFrameFromIterator(handles.rbegin()));
  if (found) returnValue = getFrameFromIterator(handles.begin());

  for (HandleBaseVectorIterator it(handles.begin()); it != handles.end() && !found; ++it) {
    returnValue = getFrameFromIterator<HandleBaseVectorIterator>(it);
    found = currentFrame < returnValue;
  }
  return returnValue;
}

void CurveGraphicsItem::signalChanged() {
  if (comp) {
    emit changed(comp->add(), curve.clone(), type, inputId);
  }
}

void CurveGraphicsItem::rebuildHandles() {
  if (type == CurveGraphicsItem::Stabilization || type == CurveGraphicsItem::StabilizationPitch ||
      type == CurveGraphicsItem::StabilizationRoll || type == CurveGraphicsItem::StabilizationYaw) {
    return;
  }
  for (HandleBaseVectorIterator it = handles.begin(); it != handles.end(); ++it) {
    delete *it;
  }
  handles.clear();
  delete constantHandle;
  constantHandle = nullptr;

  VideoStitch::Core::Spline* spline = curve.splines();
  if (spline) {
    handles.push_back(new Handle(this, timeline, spline));
    handles.back()->setZValue(1.0);
    for (spline = spline->next; spline != NULL; spline = spline->next) {
      handles.push_back(new Handle(this, timeline, spline));
      handles.back()->setZValue(1.0);
    }
  } else {
    constantHandle = new ConstantCurveHandle(this, timeline, &curve);
    constantHandle->setZValue(1.0);
  }
}

double CurveGraphicsItem::expBase10(double x) const { return exp(x * log(10.0)); }

double CurveGraphicsItem::toLogScale(double x) const { return log10(x); }

double CurveGraphicsItem::fromLogScale(double y) const { return expBase10(y); }

template <typename T>
int CurveGraphicsItem::getFrameFromIterator(T iterator) const {
  int returnValue(0);
  if (*iterator) returnValue = (*iterator)->getPoint().t;
  return returnValue;
}

void CurveGraphicsItem::rebuildPath() {
  QPainterPath path;

  const VideoStitch::Core::Spline* spline = curve.splines();
  if (!spline) {
    // Curve with no keyframes.
    const double constVal = toSceneV(curve.at(0));
    path.moveTo(0, constVal);
    path.lineTo(timeline->mapToPosition(parentView->maximum()), constVal);
  } else {
    // Extend using a constant value.
    if (spline->end.t > 0) {
      path.moveTo(0, toSceneV(spline->end.v));
      path.lineTo(timeline->mapToPosition(spline->end.t), toSceneV(spline->end.v));
    } else {
      path.moveTo(timeline->mapToPosition(spline->end.t), toSceneV(spline->end.v));
    }
    const VideoStitch::Core::Spline* lastSpline = spline;
    for (spline = spline->next; spline != NULL; spline = spline->next) {
      switch (spline->getType()) {
        case VideoStitch::Core::Spline::PointType:
          // Will never happen
          break;
        case VideoStitch::Core::Spline::LineType:
          path.lineTo(timeline->mapToPosition(spline->end.t), toSceneV(spline->end.v));
          break;
        case VideoStitch::Core::Spline::CubicType:
          /*
            full conversion matrix (inverse bezier * catmull-rom):
                0,    1,    0,     0,
              -1/6,    1,  1/6,     0,
                0,  1/6,    1,  -1/6,
                0,    0,    1,     0

            conversion doesn't require full matrix multiplication,
            so below we simplify
          */
          qreal prevFarX, prevFarY;
          if (spline->prev->prev != NULL) {
            prevFarX = spline->prev->prev->end.t;
            prevFarY = spline->prev->prev->end.v;
          } else {
            prevFarX = spline->prev->end.t - 100;  // well, we have to start somewhere...
            prevFarY = spline->prev->end.v;
          }
          qreal nextX, nextY;
          if (spline->next != NULL) {
            nextX = spline->next->end.t;
            nextY = spline->next->end.v;
          } else {
            nextX = spline->end.t + 100;  // well, we have to end somewhere...
            nextY = spline->end.v;
          }
          QPointF control1(prevFarX / qreal(-6) + spline->prev->end.t + spline->end.t / qreal(6),
                           prevFarY / qreal(-6) + spline->prev->end.v + spline->end.v / qreal(6));
          QPointF control2(spline->prev->end.t / qreal(6) + spline->end.t + nextX / qreal(-6),
                           spline->prev->end.v * qreal(0.167) + spline->end.v + nextY / qreal(-6));
          path.cubicTo(timeline->mapToPosition(control1.x()), toSceneV(control1.y()),
                       timeline->mapToPosition(control2.x()), toSceneV(control2.y()),
                       timeline->mapToPosition(spline->end.t), toSceneV(spline->end.v));
          break;
      }
      lastSpline = spline;
    }
    // Extend using a constant value.
    path.lineTo(timeline->mapToPosition(parentView->maximum()), toSceneV(lastSpline->end.v));
  }
  pathItem.setPath(path);
}
