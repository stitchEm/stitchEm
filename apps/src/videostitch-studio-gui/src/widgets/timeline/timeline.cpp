// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "timeline.hpp"

#include "curvegraphicsitem.hpp"

#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

#include <QGraphicsSceneMouseEvent>
#include <QMenu>
#include <QScrollBar>
#include <QResizeEvent>

#include <algorithm>

// As a proportion of the scene's height.
static const int MIN_OF_VISIBLE_FRAMES(10);

TimelineCursor::TimelineCursor() {
  // Cursor line.
  setOpacity(0.01);
  QGraphicsLineItem *line = new QGraphicsLineItem(0, 0, 0, SCENE_HEIGHT);
  line->setPen(QPen(Qt::red, 0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  addToGroup(line);
  // Top and bottom handles.
  setCursor(QCursor(Qt::OpenHandCursor));
  setBoundingRegionGranularity(1.0);
}

QRectF TimelineCursor::boundingRect() const {
  const Timeline *timeline = qobject_cast<Timeline *>(scene()->views().first());
  if (!timeline) {
    return QRectF();
  }
  static const double boundingRectWidth = 10;
  const double scale = timeline->visibleRect().width() / timeline->scene()->width();
  return QRectF(QPointF(-boundingRectWidth * scale, 0),
                QPointF(boundingRectWidth * scale, timeline->scene()->height()));
}

void TimelineCursor::mousePressEvent(QGraphicsSceneMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    setCursor(QCursor(Qt::ClosedHandCursor));
    event->accept();
  }
}

void TimelineCursor::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
  QPointF newPos = event->scenePos();
  QRectF constraintBounds = getConstraintBounds();
  if (!constraintBounds.contains(newPos)) {
    // Keep the item inside the scene rect.
    newPos.setX(qBound(constraintBounds.left(), newPos.x(), constraintBounds.right()));
  }

  emit newPosition(newPos);
}

void TimelineCursor::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
  setCursor(QCursor(Qt::OpenHandCursor));
  ConstrainedGraphicsItemGroup::mouseReleaseEvent(event);
}

/**
 * A non-antialiased rectangle for borders.
 */
class NoAARect : public QGraphicsRectItem {
 public:
  explicit NoAARect(const QRectF &rect) : QGraphicsRectItem(rect) { setPen(QPen(QColor(33, 33, 33), 0)); }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) {
    QPainter::RenderHints hintsBackup = painter->renderHints();
    painter->setRenderHint(QPainter::Antialiasing, false);
    QGraphicsRectItem::paint(painter, option, widget);
    painter->setRenderHints(hintsBackup, true);
  }
};

Timeline::Timeline(QWidget *parent)
    : QGraphicsView(parent),
      project(nullptr),
      timelineRect(new NoAARect(QRectF(SCENE_HBORDER, 0, TIMELINE_WIDTH, TIMELINE_HEIGHT))),
      cursor(new TimelineCursor()),
      minFrame(0),
      maxFrame(100),
      lowerBound(0),
      upperBound(0),
      sliderHeld(false),
      cursorLineColor(Qt::red),
      workingZoneColor(QColor(110, 110, 110)),
      baseOffset(0.),
      zoomMultiplier(1.),
      zoomPercentage(1),
      timelineScene(new TimelineScene(this)),
      mousePressed(false) {
  setScene(timelineScene);
  scene()->setBackgroundBrush(QBrush(QColor(55, 55, 55), Qt::SolidPattern));

  setAcceptDrops(true);
  timelineRect->setPen(Qt::NoPen);
  timelineRect->setZValue(0.9);
  scene()->addItem(timelineRect);
  scene()->setSceneRect(0, 0, SCENE_WIDTH, SCENE_HEIGHT);

  setInteractive(true);
  cursor->setConstraint(timelineRect->rect());
  cursor->setPos(SCENE_HBORDER, 0);
  cursor->setZValue(0.8);
  scene()->addItem(cursor);

  connect(cursor, SIGNAL(newPosition(QPointF)), this, SLOT(moveCursorTo(QPointF)));
  setMouseTracking(true);
  viewport()->setMouseTracking(true);

  setAttribute(Qt::WA_NoSystemBackground);
  viewport()->setAttribute(Qt::WA_NoSystemBackground);
  setAttribute(Qt::WA_TranslucentBackground, false);
  setOptimizationFlags(QGraphicsView::DontSavePainterState);
  // Everything that makes the timeline content change should call update() in order to trigger the paintEvent.
  setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
  setBackgroundBrush(QColor(0, 0, 0, 0));
  setDragMode(QGraphicsView::RubberBandDrag);
  connect(horizontalScrollBar(), SIGNAL(sliderPressed()), this, SLOT(sliderGrabbed()));
  connect(horizontalScrollBar(), SIGNAL(sliderReleased()), this, SLOT(sliderReleased()));
  connect(horizontalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(update()));
  connect(verticalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(update()));
  connect(this, SIGNAL(reqRemoveSelectedKeyframes()), this, SLOT(removeSelectedKeyframes()), Qt::QueuedConnection);
  timelineScene->addToExclusionList(timelineRect);
}

Timeline::~Timeline() {
  for (NamedItems::iterator it = curves.begin(); it != curves.end(); ++it) {
    delete it->second;
  }
}

void Timeline::resetRange() { setRange(0, 100); }

qint64 Timeline::minimum() const { return minFrame; }

qint64 Timeline::maximum() const { return maxFrame; }

qint64 Timeline::value() const { return mapFromPosition(cursor->pos().x()); }

double Timeline::mapToPosition(qint64 frame) const {
  if (frame == minFrame || frame == 0) {
    return SCENE_HBORDER;
  } else if (maxFrame == minFrame) {
    return SCENE_WIDTH - SCENE_HBORDER;
  }
  const double ratio = double(frame - minFrame) / double(maxFrame - minFrame);
  return SCENE_HBORDER + TIMELINE_WIDTH * ratio;
}

int Timeline::mapFromPosition(double position) const {
  const double ratio = (position - SCENE_HBORDER) / TIMELINE_WIDTH;
  return qRound(double(maxFrame - minFrame) * ratio) + minFrame;
}

void Timeline::setCursorLineColor(QColor color) { cursorLineColor = color; }

QColor Timeline::getCursorLineColor() const { return cursorLineColor; }

void Timeline::setWorkingZoneColor(QColor color) { workingZoneColor = color; }

QColor Timeline::getWorkingZoneColor() const { return workingZoneColor; }

QColor Timeline::getZeroLevelColor() const { return zeroLevelColor; }

void Timeline::setZeroLevelColor(QColor color) { zeroLevelColor = color; }

void Timeline::moveCursorTo(frameid_t frame) {
  frameid_t fr = qBound(minFrame, frame, maxFrame);
  emit reqSeek(fr);
}

void Timeline::moveCursorTo(const QPointF &newPos) { moveCursorTo(mapFromPosition(newPos.x())); }

void Timeline::setRange(frameid_t min, frameid_t max) {
  if (minFrame == min && maxFrame == max) {
    return;
  }
  minFrame = min;
  maxFrame = max;
  if (maxFrame == NO_LAST_FRAME) {
    maxFrame = minFrame;
  }
  updateSliders();
  update();
}

void Timeline::setValue(frameid_t frame) {
  if (cursor) {
    qint64 boundFrame = qBound(minFrame, frame, maxFrame);
    cursor->setPos(mapToPosition(boundFrame), 0);
    update();
  }
}

void Timeline::setBounds(qint64 min, qint64 max) {
  if (lowerBound == min && upperBound == max) {
    return;
  }
  lowerBound = min;
  upperBound = max;
  update();
}

bool Timeline::addNamedCurve(const TimelineItemPayload &payload) {
  if (curves.find(payload) != curves.end()) {
    return false;
  }
  CurveGraphicsItem *curveItem = new CurveGraphicsItem(this, *payload.curve, payload.curveType, payload.minCurveValue,
                                                       payload.maxCurveValue, payload.inputId, payload.color);
  qRegisterMetaType<CurveGraphicsItem::Type>("CurveGraphicsItem::Type");
  connect(curveItem, SIGNAL(changed(SignalCompressionCaps *, VideoStitch::Core::Curve *, CurveGraphicsItem::Type, int)),
          this,
          SLOT(on_curve_changed(SignalCompressionCaps *, VideoStitch::Core::Curve *, CurveGraphicsItem::Type, int)));
  connect(this, SIGNAL(reqRefreshCurves()), curveItem, SLOT(rebuildPath()));
  curveItem->setZValue(2);
  curves[payload] = curveItem;
  update();
  emit reqUpdateKeyFrames(hasVisibleKeyFrames());
  return true;
}

void Timeline::on_curve_changed(SignalCompressionCaps *comp, VideoStitch::Core::Curve *curve,
                                CurveGraphicsItem::Type type, int inputId) {
  emit reqCurveChanged(comp, curve, type, inputId);
}

void Timeline::on_quaternion_curve_changed(SignalCompressionCaps *comp, VideoStitch::Core::QuaternionCurve *curve,
                                           CurveGraphicsItem::Type type, int inputId) {
  emit reqQuaternionCurveChanged(comp, curve, type, inputId);
}

void Timeline::removeNamedCurve(const TimelineItemPayload &payload) {
  NamedItems::iterator it = curves.find(payload);
  timelineScene->clearExcludedFromSelectionList();
  if (it == curves.end()) {
    return;
  }
  delete it->second;
  curves.erase(it);
  update();
  emit reqUpdateKeyFrames(hasVisibleKeyFrames());
}

void Timeline::removeSelectedKeyframesAsync() { emit reqRemoveSelectedKeyframes(); }

bool Timeline::hasVisibleKeyFrames() const {
  for (NamedItems::const_iterator it = curves.begin(); it != curves.end(); ++it) {
    if (it->second->hasKeyframe()) {
      return true;
    }
  }
  return false;
}

void Timeline::restoreScale() {
  if (visibleRect().width() > 0) {
    double backFactor = SCENE_WIDTH / visibleRect().width();
    scale(1.0 / backFactor, 1.0);
  }
}

QString Timeline::getFramestringFromFrame(frameid_t frame) const {
  return TimeConverter::frameToTimeDisplay(frame, GlobalController::getInstance().getController()->getFrameRate());
}

int Timeline::getCursorPosition() const { return mapFromScene(cursor->pos().x(), 0).x(); }

int Timeline::getLowerBoundPosition() const { return mapFromScene(mapToPosition(lowerBound), 0).x(); }

int Timeline::getUpperBoundPosition() const { return mapFromScene(mapToPosition(upperBound), 0).x(); }

void Timeline::clearTimeline() { emit reqResetCurves(); }

int Timeline::nextKeyFrame() const {
  int nextKeyFrame(value());
  for (NamedItems::const_iterator it = curves.begin(); it != curves.end(); ++it) {
    if (it->second->getType() < CurveGraphicsItem::Type::StabilizationYaw ||
        it->second->getType() > CurveGraphicsItem::Type::StabilizationRoll) {
      nextKeyFrame = it->second->getNextKeyFrame(value());
    }
  }
  return nextKeyFrame;
}

int Timeline::prevKeyFrame() const {
  int prevKeyFrame(value());
  const bool notZero(value() != 0);
  for (NamedItems::const_iterator it = curves.begin(); it != curves.end() && notZero; ++it) {
    if (it->second->getType() < CurveGraphicsItem::Type::StabilizationYaw ||
        it->second->getType() > CurveGraphicsItem::Type::StabilizationRoll) {
      prevKeyFrame = it->second->getPrevKeyFrame(value());
    }
  }
  return prevKeyFrame;
}

void Timeline::moveCenterOfScrollbar(double proportion, QScrollBar *bar) {
  bar->setValue((bar->maximum() + bar->pageStep()) * proportion - bar->pageStep() / 2.0);
  update();
}

double Timeline::getMiddlePagestepPosition(QScrollBar *bar) {
  return (double)(bar->value() + bar->pageStep() / 2.0) / (double)(bar->maximum() + bar->pageStep());
}

void Timeline::resizeEvent(QResizeEvent *event) {
  double hProportion = getMiddlePagestepPosition(horizontalScrollBar());
  double vProportion = getMiddlePagestepPosition(verticalScrollBar());
  QGraphicsView::resizeEvent(event);
  if (visibleRect().contains(scene()->sceneRect())) {
    fillViewport();
  }
  moveCenterOfScrollbar(hProportion, horizontalScrollBar());
  moveCenterOfScrollbar(vProportion, verticalScrollBar());
}

void Timeline::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  resetTimeline();
}

void Timeline::drawBackground(QPainter *painter, const QRectF &rect) {
  QGraphicsView::drawBackground(painter, rect);

  if (lowerBound != upperBound) {
    painter->setBrush(workingZoneColor);
    painter->setPen(Qt::NoPen);
    painter->drawRect(
        QRectF(QPointF(mapToPosition(lowerBound), rect.top()), QPointF(mapToPosition(upperBound), rect.bottom())));
  } else {
    painter->setPen(QPen(workingZoneColor, 0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter->drawLine(mapToPosition(lowerBound), rect.top(), mapToPosition(lowerBound), rect.bottom());
  }
  painter->setPen(QPen(zeroLevelColor, 0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  painter->drawLine(-scene()->sceneRect().width(), scene()->sceneRect().height() / 2.0,
                    scene()->sceneRect().width() * 2, scene()->sceneRect().height() / 2.0);
}

void Timeline::paintEvent(QPaintEvent *event) {
  QGraphicsView::paintEvent(event);
  QPainter painter(viewport());
  painter.setPen(QPen(getCursorLineColor(), 0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  painter.drawLine(getCursorPosition(), 0, getCursorPosition(), height());
  emit reqRefreshTicks();
}

void Timeline::setAnchor(QGraphicsView::ViewportAnchor anchor) {
  setTransformationAnchor(anchor);
  setResizeAnchor(anchor);
}

void Timeline::fillViewport() {
  fitInView(scene()->sceneRect(), Qt::IgnoreAspectRatio);
  updateSliders();
}

void Timeline::updateSliders() {
  emit reqUpdateZoomSliders(scene()->sceneRect().width() /
                            (mapToPosition(minFrame + MIN_OF_VISIBLE_FRAMES) - mapToPosition(minFrame)));
}

void Timeline::mouseMoveEvent(QMouseEvent *event) {
  QGraphicsView::mouseMoveEvent(event);
  QSet<QGraphicsItem *> toSelectBack;
  QSet<QGraphicsItem *> toUnselect;
  if (event->modifiers() == Qt::AltModifier && mousePressed) {
    QRectF selectionArea = QRectF(mapToScene(antiSelectionStart), mapToScene(event->pos())).normalized();
    toUnselect = scene()->items(selectionArea).toSet();
    // The oposite region of the selection is the sum of 4 rectangles
    QRectF top, bottom, left, right;
    top = QRectF(scene()->sceneRect().topLeft(), QPointF(selectionArea.topRight().x(), selectionArea.topRight().y()))
              .normalized();
    bottom = QRectF(scene()->sceneRect().bottomRight(),
                    QPointF(selectionArea.bottomLeft().x(), selectionArea.bottomLeft().y()))
                 .normalized();
    left = QRectF(scene()->sceneRect().bottomLeft(), QPointF(selectionArea.topLeft().x(), selectionArea.topLeft().y()))
               .normalized();
    right = QRectF(scene()->sceneRect().topRight(),
                   QPointF(selectionArea.bottomRight().x(), selectionArea.bottomRight().y()))
                .normalized();

    toSelectBack = scene()->items(top).toSet();
    toSelectBack = toSelectBack.unite(scene()->items(bottom).toSet());
    toSelectBack = toSelectBack.unite(scene()->items(left).toSet());
    toSelectBack = toSelectBack.unite(scene()->items(right).toSet());

    timelineScene->addToUnselectionSet(toUnselect);
    timelineScene->removeToUnselectionSet(toSelectBack);
  }

  if (event->modifiers() == Qt::ShiftModifier || event->modifiers() == Qt::AltModifier) {
    foreach (QGraphicsItem *item, selectedItems) {
      item->setSelected(!timelineScene->getExcludedFromSelection().contains(item));
    }
    foreach (QGraphicsItem *item, timelineScene->getExcludedFromSelection()) { item->setSelected(false); }
    if (event->modifiers() == Qt::AltModifier && mousePressed) {
      QSet<QGraphicsItem *> borderItems = toSelectBack;
      borderItems = borderItems.intersect(toUnselect);
      foreach (QGraphicsItem *item, borderItems) { item->setSelected(false); }
    }
  }
}

void Timeline::mouseReleaseEvent(QMouseEvent *event) {
  mousePressed = false;
  selectedBeforeMove.clear();
  timelineScene->clearExcludedFromSelectionList();
  viewport()->setCursor(QCursor(Qt::ArrowCursor));
  selectedItems = scene()->selectedItems();
  QGraphicsView::mouseReleaseEvent(event);
  foreach (QGraphicsItem *item, selectedItems) { item->setSelected(true); }
  if (!event->isAccepted()) {
    event->ignore();
    return;
  }
  if (event->button() == Qt::RightButton) {
    QMenu menu;
    QAction *addKeyframe = nullptr;
    for (NamedItems::const_iterator it = curves.begin(); it != curves.end(); ++it) {
      if (it->second->getType() > CurveGraphicsItem::Type::GlobalOrientation && project &&
          !project->hasImagesOrProceduralsOnly()) {
        addKeyframe = menu.addAction(tr("Add keyframe"));
        break;
      }
    }
    QAction *removeSelected = nullptr;
    if (!scene()->selectedItems().isEmpty()) {
      removeSelected = menu.addAction(tr("Remove selected keyframes"));
    }

    QAction *remove = menu.addAction(tr("Clear timeline"));

    QAction *selectedAction = menu.exec(event->globalPos());
    if (selectedAction) {
      if (selectedAction == addKeyframe) {
        addKeyframeHere();
      } else if (selectedAction == remove) {
        auto result = MsgBoxHandler::getInstance()->genericSync(
            tr("Are you sure you want to delete all the keyframes?"), tr("Deletion confirmation"),
            QString(WARNING_ICON), QMessageBox::Yes | QMessageBox::No);
        if (result == QMessageBox::Yes) {
          clearTimeline();
        }
      } else if (selectedAction == removeSelected) {
        removeSelectedKeyframes();
      }
    }
  }
}

void Timeline::keyPressEvent(QKeyEvent *event) {
  if ((event->key() == Qt::Key_Backspace || event->key() == Qt::Key_Delete) && !scene()->selectedItems().isEmpty()) {
    removeSelectedKeyframes();
  }
}

void Timeline::mousePressEvent(QMouseEvent *event) {
  QGraphicsItem *itemUderMouse = itemAt(event->pos());

  if (itemUderMouse == timelineRect && event->modifiers() == Qt::NoModifier) {
    QGraphicsView::mousePressEvent(event);
  } else {
    if (event->modifiers() == Qt::AltModifier) {
      mousePressed = true;
      antiSelectionStart = event->pos();
      selectedBeforeMove = scene()->selectedItems().toSet();
    }
    selectedItems = scene()->selectedItems();
    bool previouslySelected = itemUderMouse != NULL && itemUderMouse->isSelected();
    QGraphicsView::mousePressEvent(event);
    if (event->modifiers() == Qt::ShiftModifier || event->modifiers() == Qt::AltModifier ||
        (previouslySelected && selectedItems.size() > 1) || event->button() == Qt::RightButton) {
      foreach (QGraphicsItem *item, selectedItems) {
        item->setSelected(!timelineScene->getExcludedFromSelection().contains(item));
      }
    }

    if (event->isAccepted()) {
      event->ignore();
      return;
    }
  }
}

void Timeline::wheelEvent(QWheelEvent *event) {
  float numDegrees = event->delta() / 10.0f;
  float stepSize = numDegrees / 15.0f;
  float factor = 1.0f + stepSize / 5.0f;
  if (factor < 0.0f) {
    factor = fabs(factor);
  }

  setAnchor(QGraphicsView::AnchorViewCenter);

  // do not zoom out if we already see all the height
  if (visibleRect().top() <= 0 && visibleRect().bottom() >= scene()->sceneRect().bottom() && factor < 1.0f) {
    return;
  }

  // do not zoom if we zoomed already too much
  if (fabs(visibleRect().bottom() - visibleRect().top()) <= (scene()->height() / 100000) && factor > 1.0f) {
    return;
  }

  scale(1.0, factor);

  // if we zoom out too much
  while (visibleRect().top() < 0 && visibleRect().bottom() > scene()->sceneRect().bottom()) {
    double backFactor = (visibleRect().bottom() - visibleRect().top()) / scene()->height();
    if (backFactor >= 2.0) {
      backFactor = 1.99;
    }
    scale(1.0, backFactor);
  }
}

QRectF Timeline::visibleRect() const { return mapToScene(viewport()->geometry()).boundingRect(); }

void Timeline::sliderGrabbed() { sliderHeld = true; }

void Timeline::sliderReleased() { sliderHeld = false; }

void Timeline::setZoomLevel(int percentage) {
  zoomPercentage = percentage;
  QMatrix scaleMatrix;
  int vSliderValue = verticalScrollBar()->value();
  scaleMatrix.scale(exp((double)percentage / zoomMultiplier), matrix().m22());
  setMatrix(scaleMatrix);
  if (visibleRect().left() <= 0 && visibleRect().right() >= SCENE_WIDTH) {
    restoreScale();
  }
  centerOn(cursor);
  verticalScrollBar()->setValue(vSliderValue);
  update();
}

void Timeline::addKeyframeHere() {
  // Add a keyframe for all visible curves.
  for (NamedItems::const_iterator it = curves.begin(); it != curves.end(); ++it) {
    it->second->addKeyframe(value());
  }
}

void Timeline::removeSelectedKeyframes() {
  QList<CurveGraphicsItem *> selectedCurves;
  foreach (QGraphicsItem *item, scene()->selectedItems()) {
    CurveGraphicsItem *curve = static_cast<CurveGraphicsItem *>(item->parentItem());
    curve->addToRemovalList(mapFromPosition(item->pos().x()));
    if (selectedCurves.indexOf(curve) == -1) {
      selectedCurves.push_back(curve);
    }
  }

  foreach (CurveGraphicsItem *curve, selectedCurves) {
    curve->processRemovalList();
    if (!curve->CurveGraphicsItem::hasKeyframe()) {
      emit reqResetCurve(curve->getType(), curve->getInputId());
    }
  }
  timelineScene->clearExcludedFromSelectionList();
  scene()->clearSelection();
}

void Timeline::resetTimeline() {
  setRange(minFrame, maxFrame);
  setBounds(lowerBound, upperBound);
  fillViewport();
  restoreScale();
  update();
}

void Timeline::setProject(ProjectDefinition *p) { project = p; }

void Timeline::clearSelectedItems() { selectedItems.clear(); }

void Timeline::setZoomMultiplier(double mult) { zoomMultiplier = mult; }

double Timeline::getScaleValue() const { return zoomPercentage; }
