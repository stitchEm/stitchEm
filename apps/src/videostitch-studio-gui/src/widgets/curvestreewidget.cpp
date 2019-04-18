// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "curvestreewidget.hpp"

#include "../videostitcher/postprodprojectdefinition.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"

#include "libvideostitch/inputDef.hpp"

#include <QMenu>
#include <QAction>
#include <QTreeWidgetItem>
#include <QMouseEvent>

#define ORIENTATION_STRING CurvesTreeWidget::tr("Orientation")
#define GLOB_ORIENTATION_STRING CurvesTreeWidget::tr("Global orientation")
#define STAB_STRING CurvesTreeWidget::tr("Stabilization")
#define EV_STRING CurvesTreeWidget::tr("Exposure compensation")
#define GLOBAL_STRING CurvesTreeWidget::tr("Global")
#define INPUT_STRING CurvesTreeWidget::tr("Input %0")
#define RED_CORR_STRING CurvesTreeWidget::tr("Red correction")
#define BLUE_CORR_STRING CurvesTreeWidget::tr("Blue correction")
#define EV_STRING_MIN CurvesTreeWidget::tr("Exposure")
#define YAW_STRING CurvesTreeWidget::tr("Yaw")
#define ROLL_STRING CurvesTreeWidget::tr("Roll")
#define PITCH_STRING CurvesTreeWidget::tr("Pitch")

#define MIN_EV -10.0
#define MAX_EV 10.0
// FIXME: update that when the WB stabilization will return good values.

#define MIN_WB pow(10.0, -38.0)
#define MAX_WB pow(10.0, 38.0)

#define MIN_ROT -180.0
#define MAX_ROT 180.0

#define MIN_OR -1.0
#define MAX_OR 1.0

const QColor CurvesTreeWidget::STABILIZED_YAW_COLOR = QColor(255, 222, 0);
const QColor CurvesTreeWidget::STABILIZED_PITCH_COLOR = QColor(0, 69, 255);
const QColor CurvesTreeWidget::STABILIZED_ROLL_COLOR = QColor(24, 171, 0);
const QColor CurvesTreeWidget::GLOBAL_EV_COLOR = QColor(255, 150, 50);
const QColor CurvesTreeWidget::CR_COLOR = QColor(255, 69, 0);
const QColor CurvesTreeWidget::CB_COLOR = QColor(0, 69, 255);

CurvesTreeWidget::CurvesTreeWidget(QWidget* parent) : QTreeWidget(parent), timeline(nullptr) {
  connect(this, SIGNAL(itemSelectionChanged()), this, SLOT(updateTimelineContents()));
  connect(this, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(onItemExpanded(QTreeWidgetItem*)));
  connect(this, SIGNAL(itemCollapsed(QTreeWidgetItem*)), this, SLOT(onItemCollapsed(QTreeWidgetItem*)));
  connect(this, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(onItemClicked(QTreeWidgetItem*)));
  StateManager::getInstance()->registerObject(this);
}

CurvesTreeWidget::~CurvesTreeWidget() { clear(); }

void CurvesTreeWidget::clear() {
  for (size_t i = 0; i < garbageToCollect.size(); ++i) {
    delete garbageToCollect[i];
  }
  garbageToCollect.clear();
  QTreeWidget::clear();
}

void CurvesTreeWidget::addCurveChild(QTreeWidgetItem* parent, const QString& name,
                                     const VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type curveType,
                                     double minValue, double maxValue, int inputId, const QColor& color) {
  QTreeWidgetItem* item = new QTreeWidgetItem(QStringList(name));
  if (curve != NULL) {
    VideoStitch::Core::Curve* curveCopy(curve->clone());
    garbageToCollect.push_back(curveCopy);
    item->setData(
        0, Qt::UserRole,
        QVariant::fromValue(TimelineItemPayload(name, curveCopy, curveType, minValue, maxValue, inputId, color)));
    item->setData(0, Qt::DecorationRole, color);
  }
  parent->addChild(item);
  parent->QTreeWidgetItem::sortChildren(0, Qt::DescendingOrder);
}

void CurvesTreeWidget::populate(const PostProdProjectDefinition& project) {
  QList<QStringList> itemsToReactivate;
  foreach (QTreeWidgetItem* item, previouslySelectedItems) {
    QTreeWidgetItem* parent = item;
    QStringList path;
    path.push_front(parent->text(0));
    while ((parent = parent->parent()) != NULL) {
      path.push_back(parent->text(0));
    }
    itemsToReactivate << path;
  }
  clear();
  previouslySelectedItems.clear();
  userSelectedItems.clear();
  QTreeWidgetItem* orientationCurves = new QTreeWidgetItem(QStringList(ORIENTATION_STRING));
  VideoStitch::Core::Curve *yaw, *pitch, *roll;
  VideoStitch::Core::toEuler(project.getPanoConst()->getGlobalOrientation(), &yaw, &pitch, &roll);
  addCurveChild(orientationCurves, GLOB_ORIENTATION_STRING, yaw, CurveGraphicsItem::GlobalOrientation, MIN_ROT, MAX_ROT,
                -1, STABILIZED_YAW_COLOR);
  delete yaw;
  delete pitch;
  delete roll;

  if (!project.hasImagesOrProceduralsOnly()) {
    QTreeWidgetItem* stabCurves = new QTreeWidgetItem(QStringList(STAB_STRING));
    addCurveChild(stabCurves, YAW_STRING, &project.getPanoConst()->getStabilizationYaw(),
                  CurveGraphicsItem::StabilizationYaw, MIN_ROT, MAX_ROT, -1, STABILIZED_YAW_COLOR);
    addCurveChild(stabCurves, PITCH_STRING, &project.getPanoConst()->getStabilizationPitch(),
                  CurveGraphicsItem::StabilizationPitch, MIN_ROT, MAX_ROT, -1, STABILIZED_PITCH_COLOR);
    addCurveChild(stabCurves, ROLL_STRING, &project.getPanoConst()->getStabilizationRoll(),
                  CurveGraphicsItem::StabilizationRoll, MIN_ROT, MAX_ROT, -1, STABILIZED_ROLL_COLOR);
    addTopLevelItem(stabCurves);
  }
  addTopLevelItem(orientationCurves);
  // Input exposure curves.
  {
    QTreeWidgetItem* exposureCurves = new QTreeWidgetItem(QStringList(EV_STRING));
    exposureCurves->setFlags(Qt::ItemIsEnabled);
    QTreeWidgetItem* global = new QTreeWidgetItem(QStringList(QString(GLOBAL_STRING)));
    global->setFlags(Qt::ItemIsEnabled);
    addCurveChild(global, RED_CORR_STRING, &project.getPanoConst()->getRedCB(), CurveGraphicsItem::GlobalRedCorrection,
                  MIN_WB, MAX_WB, -1, CR_COLOR);
    addCurveChild(global, EV_STRING_MIN, &project.getPanoConst()->getExposureValue(), CurveGraphicsItem::GlobalExposure,
                  MIN_EV, MAX_EV, -1, GLOBAL_EV_COLOR);
    addCurveChild(global, BLUE_CORR_STRING, &project.getPanoConst()->getBlueCB(),
                  CurveGraphicsItem::GlobalBlueCorrection, MIN_WB, MAX_WB, -1, CB_COLOR);
    exposureCurves->addChild(global);
    for (int i = 0; i < (int)project.getNumInputs(); ++i) {
      QTreeWidgetItem* input = new QTreeWidgetItem(QStringList(INPUT_STRING.arg(i)));
      input->setFlags(Qt::ItemIsEnabled);

      QColor itemColor = QColor::fromHsl((360 * i) / (int)project.getNumInputs(), 255, 96);
      addCurveChild(input, RED_CORR_STRING, &project.getPanoConst()->getInput(i).getRedCB(),
                    CurveGraphicsItem::RedCorrection, MIN_WB, MAX_WB, i, CR_COLOR);
      addCurveChild(input, EV_STRING_MIN, &project.getPanoConst()->getInput(i).getExposureValue(),
                    CurveGraphicsItem::InputExposure, MIN_EV, MAX_EV, i, itemColor);
      addCurveChild(input, BLUE_CORR_STRING, &project.getPanoConst()->getInput(i).getBlueCB(),
                    CurveGraphicsItem::BlueCorrection, MIN_WB, MAX_WB, i, CB_COLOR);
      exposureCurves->addChild(input);
    }
    addTopLevelItem(exposureCurves);
  }

  setAnimated(false);
  foreach (QStringList path, itemsToReactivate) {
    QStringListIterator it(path);
    QString itemString = it.next();
    QList<QTreeWidgetItem*> potentialMatches = findItems(itemString, Qt::MatchRecursive, 0);
    int level = 1;
    while (it.hasNext() && potentialMatches.size() > 1) {
      QString parent = it.next();
      QList<QTreeWidgetItem*> potentialMatchesCopy = potentialMatches;
      foreach (QTreeWidgetItem* potentialMatch, potentialMatchesCopy) {
        QTreeWidgetItem* parentItem = potentialMatch;

        for (int i = 0; i < level; i++) {
          parentItem = potentialMatch->parent();
        }
        if (parentItem->text(0) != parent) {
          potentialMatches.removeAll(potentialMatch);
        }
      }
      level++;
    }
    if (potentialMatches.size() == 1) {
      potentialMatches.first()->setSelected(true);
      userSelectedItems.push_back(potentialMatches.first());
      QTreeWidgetItem* parentItem = potentialMatches.first();
      while ((parentItem = parentItem->parent())) {
        parentItem->setExpanded(true);
      }
    }
  }
  setAnimated(true);

  updateGeometry();
}

void CurvesTreeWidget::onProjectOrientable(bool orientable) {
  QTreeWidgetItem* orientationCurves = findItems(ORIENTATION_STRING, Qt::MatchRecursive, 0).first();
  QTreeWidgetItem* stabCurves = findItems(STAB_STRING, Qt::MatchRecursive, 0).value(0);
  if (orientable) {
    orientationCurves->setFlags(Qt::ItemIsEnabled);
    if (stabCurves) {
      stabCurves->setFlags(Qt::ItemIsEnabled);
    }
  } else {
    orientationCurves->setFlags(Qt::NoItemFlags);
    if (stabCurves) {
      stabCurves->setFlags(Qt::NoItemFlags);
    }
  }
  orientationCurves->setDisabled(!orientable);
}

void CurvesTreeWidget::updateTimelineContents() {
  sender();
  if (timeline == NULL) {
    return;
  }
  const QList<QTreeWidgetItem*>& selItems = selectedItems();
  // Add all newly selected items.
  for (QList<QTreeWidgetItem*>::const_iterator it = selItems.begin(); it != selItems.end(); ++it) {
    QSet<QTreeWidgetItem*>::iterator prevIt = previouslySelectedItems.find(*it);
    if (prevIt == previouslySelectedItems.end()) {
      TimelineItemPayload payload((*it)->data(0, Qt::UserRole).value<TimelineItemPayload>());
      switch (payload.type) {
        case TimelineItemPayload::Type::None:
          break;
        case TimelineItemPayload::Type::Curve:
          timeline->addNamedCurve(payload);
          break;
      }
    } else {
      previouslySelectedItems.erase(prevIt);
    }
  }
  // Remove all newly deselected items.
  removePreviouslySelectedFromTimeline();
  for (QList<QTreeWidgetItem*>::const_iterator it = selItems.begin(); it != selItems.end(); ++it) {
    previouslySelectedItems.insert(*it);
  }
}

void CurvesTreeWidget::onItemExpanded(QTreeWidgetItem* item) { setChildrenSelected(item, true); }

void CurvesTreeWidget::onItemCollapsed(QTreeWidgetItem* item) { setChildrenSelected(item, false); }

void CurvesTreeWidget::onItemClicked(QTreeWidgetItem* item, bool fromEntered) {
  // This item is a leaf
  if (item->childCount() > 0) {
    item->setExpanded(!item->isExpanded());
  } else {
    bool selected = (fromEntered) ? !item->isSelected() : item->isSelected();

    if (selected) {
      userSelectedItems.push_back(item);
    } else {
      userSelectedItems.removeAll(item);
    }
  }
}

void CurvesTreeWidget::removePreviouslySelectedFromTimeline() {
  if (timeline) {
    for (QSet<QTreeWidgetItem*>::const_iterator it = previouslySelectedItems.begin();
         it != previouslySelectedItems.end(); ++it) {
      if (!(*it)->data(0, Qt::UserRole).canConvert<TimelineItemPayload>() || (*it)->data(0, Qt::UserRole).isNull() ||
          !(*it)->data(0, Qt::UserRole).isValid()) {
        return;
      }
      TimelineItemPayload payload((*it)->data(0, Qt::UserRole).value<TimelineItemPayload>());
      switch (payload.type) {
        case TimelineItemPayload::Type::None:
          break;
        case TimelineItemPayload::Type::Curve:
          emit reqRemoveNamedCurve(payload);
          break;
      }
    }
  }
  previouslySelectedItems.clear();
}

void CurvesTreeWidget::changeState(GUIStateCaps::State s) {
  switch (s) {
    case GUIStateCaps::idle:
      setDisabled(true);
      removePreviouslySelectedFromTimeline();
      clear();
      break;
    case GUIStateCaps::disabled:
    case GUIStateCaps::frozen:
      setDisabled(true);
      break;
    case GUIStateCaps::stitch:
      setEnabled(true);
      break;
    default:
      Q_ASSERT(0);
      return;
  }
}

void CurvesTreeWidget::updateCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type, int inputId) {
  switch (type) {
    case CurveGraphicsItem::InputExposure:
    case CurveGraphicsItem::RedCorrection:
    case CurveGraphicsItem::BlueCorrection:
      replaceInputCurve(curve, type, inputId);
      break;
    case CurveGraphicsItem::GlobalExposure:
    case CurveGraphicsItem::GlobalRedCorrection:
    case CurveGraphicsItem::GlobalBlueCorrection:
      replaceGlobalCurve(curve, type);
      break;
    default:
      break;
  }
}

void CurvesTreeWidget::updateQuaternionCurve(VideoStitch::Core::QuaternionCurve* curve, CurveGraphicsItem::Type type,
                                             int inputId) {
  Q_UNUSED(inputId)
  replaceGlobalQuaternionCurve(curve, type);
}

void CurvesTreeWidget::replaceInputCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type, int index) {
  if (index < 0) {
    return;
  }
  QTreeWidgetItem* input = findItems(INPUT_STRING.arg(index), Qt::MatchRecursive, 0).first();
  QTreeWidgetItem* exposureItem = findItems(EV_STRING, Qt::MatchRecursive, 0).first();
  bool isWB = true;

  QString idString;
  QColor color;
  switch (type) {
    case CurveGraphicsItem::InputExposure:
      isWB = false;
      idString = EV_STRING_MIN;
      color = QColor::fromHsl((360 * index) / (int)exposureItem->childCount(), 255, 96);
      break;
    case CurveGraphicsItem::RedCorrection:
      idString = RED_CORR_STRING;
      color = CR_COLOR;
      break;
    case CurveGraphicsItem::BlueCorrection:
      idString = BLUE_CORR_STRING;
      color = CB_COLOR;
      break;
    default:
      return;
  }

  if (!input) {
    return;
  }
  QList<QTreeWidgetItem*> inputExposureItems = findItems(idString, Qt::MatchRecursive, 0);
  if (!inputExposureItems.size()) {
    return;
  }
  std::vector<QTreeWidgetItem*> itemsToRemove;
  for (auto it = inputExposureItems.begin(); it != inputExposureItems.end(); it++) {
    QTreeWidgetItem* item = *it;
    if (!item->parent() || item->parent()->text(0) == GLOBAL_STRING) {
      itemsToRemove.push_back(item);
    }
  }

  for (QTreeWidgetItem* item : itemsToRemove) {
    inputExposureItems.removeAll(item);
  }

  int offset = 0;
  QTreeWidgetItem* inputExposure = inputExposureItems.at(index + offset);
  bool selected = inputExposure->isSelected();
  QString parentString = inputExposure->parent()->text(0);

  delete inputExposure;
  addCurveChild(input, idString, curve, type, (isWB) ? MIN_WB : MIN_EV, (isWB) ? MAX_WB : MAX_EV, index, color);
  for (QTreeWidgetItem* item : findItems(idString, Qt::MatchRecursive, 0)) {
    if (item && item->parent() && item->parent()->text(0) == parentString) {
      item->setSelected(selected);
    }
  }
}

void CurvesTreeWidget::replaceGlobalCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type) {
  QString idString;
  QColor color;
  QString rootString;
  double minVal, maxVal;
  switch (type) {
    case CurveGraphicsItem::GlobalExposure:
      idString = EV_STRING_MIN;
      color = GLOBAL_EV_COLOR;
      rootString = GLOBAL_STRING;
      break;
    case CurveGraphicsItem::GlobalBlueCorrection:
      idString = BLUE_CORR_STRING;
      color = CB_COLOR;
      rootString = GLOBAL_STRING;
      break;
    case CurveGraphicsItem::GlobalRedCorrection:
      idString = RED_CORR_STRING;
      color = CR_COLOR;
      rootString = GLOBAL_STRING;
      break;
    default:
      return;
  }

  switch (type) {
    case CurveGraphicsItem::GlobalExposure:
      minVal = MIN_EV;
      maxVal = MAX_EV;
      break;
    case CurveGraphicsItem::GlobalBlueCorrection:
    case CurveGraphicsItem::GlobalRedCorrection:
      minVal = MIN_WB;
      maxVal = MAX_WB;
      break;
    default:
      return;
  }

  if (!findItems(rootString, Qt::MatchRecursive, 0).size()) {
    return;
  }

  QTreeWidgetItem* globalCurves = findItems(rootString, Qt::MatchRecursive, 0).first();
  if (!globalCurves) {
    return;
  }
  QTreeWidgetItem* curveItem = NULL;
  foreach (QTreeWidgetItem* item, findItems(idString, Qt::MatchRecursive, 0)) {
    if (item->parent() == globalCurves) {
      curveItem = item;
      break;
    }
  }

  if (curveItem) {
    bool selected = curveItem->isSelected();
    delete curveItem;
    addCurveChild(globalCurves, idString, curve, type, minVal, maxVal, -1, color);
    foreach (QTreeWidgetItem* item, findItems(idString, Qt::MatchRecursive, 0)) {
      if (item->parent() == globalCurves) {
        item->setSelected(selected);
        break;
      }
    }
  }
}

namespace {
// Returns the maximum absolute value in the curve.
double maxCurveAbsValue(const VideoStitch::Core::Curve* curve) {
  const VideoStitch::Core::Spline* spline = curve->splines();
  if (spline) {
    double maxAbsValue = 0.0;
    for (spline = spline->next; spline != NULL; spline = spline->next) {
      maxAbsValue = std::max(maxAbsValue, spline->end.v);
    }
    return maxAbsValue;
  } else {
    return std::abs(curve->at(0));
  }
}
}  // namespace

void CurvesTreeWidget::replaceGlobalQuaternionCurve(VideoStitch::Core::QuaternionCurve* curve,
                                                    CurveGraphicsItem::Type type) {
  QString orientationString = GLOB_ORIENTATION_STRING;
  QString yString = YAW_STRING, pString = PITCH_STRING, rString = ROLL_STRING;
  QColor color = STABILIZED_YAW_COLOR;
  QString rootString = (type == CurveGraphicsItem::GlobalOrientation) ? ORIENTATION_STRING : STAB_STRING;

  double minVal, maxVal;
  VideoStitch::Core::Curve *yaw, *pitch, *roll;
  VideoStitch::Core::toEuler(*curve, &yaw, &pitch, &roll);
  if (type == CurveGraphicsItem::GlobalOrientation) {
    minVal = MIN_OR;
    maxVal = MAX_OR;
  } else {
    maxVal =
        std::max(maxCurveAbsValue(yaw), std::max(maxCurveAbsValue(pitch), std::max(maxCurveAbsValue(roll), MAX_ROT)));
    minVal = -maxVal;
  }

  if (!findItems(rootString, Qt::MatchRecursive, 0).size()) {
    return;
  }

  QTreeWidgetItem* globalCurves = findItems(rootString, Qt::MatchRecursive, 0).first();
  if (!globalCurves) {
    return;
  }

  QList<QString> itemStrings;
  if (type == CurveGraphicsItem::GlobalOrientation) {
    itemStrings << orientationString;
  } else {
    itemStrings << yString;
    itemStrings << pString;
    itemStrings << rString;
  }
  foreach (QString id, itemStrings) {
    QTreeWidgetItem* curveItem{nullptr};

    foreach (QTreeWidgetItem* item, findItems(id, Qt::MatchRecursive, 0)) {
      if (item->parent() == globalCurves) {
        curveItem = item;
        break;
      }
    }

    if (curveItem) {
      bool selected = curveItem->isSelected();
      delete curveItem;
      if (type == CurveGraphicsItem::GlobalOrientation) {
        addCurveChild(globalCurves, id, yaw, type, minVal, maxVal, -1, color);
        delete pitch;
        delete roll;
      } else {
        if (id == yString) {
          addCurveChild(globalCurves, id, yaw, type, minVal, maxVal, -1, STABILIZED_YAW_COLOR);
        } else if (id == pString) {
          addCurveChild(globalCurves, id, pitch, type, minVal, maxVal, -1, STABILIZED_PITCH_COLOR);
        } else {
          addCurveChild(globalCurves, id, roll, type, minVal, maxVal, -1, STABILIZED_ROLL_COLOR);
        }
      }
      foreach (QTreeWidgetItem* item, findItems(id, Qt::MatchRecursive, 0)) {
        if (item->parent() == globalCurves) {
          item->setSelected(selected);
          break;
        }
      }
    }
  }
}

void CurvesTreeWidget::mousePressEvent(QMouseEvent* event) {
  QTreeWidget::mousePressEvent(event);

  QTreeWidgetItem* item = itemAt(event->pos());
  if (event->button() == Qt::RightButton && item) {
    const QTreeWidgetItem* exposureItem = findItems(EV_STRING, Qt::MatchRecursive, 0).first();
    const QTreeWidgetItem* orientationItem = findItems(ORIENTATION_STRING, Qt::MatchRecursive, 0).first();
    const QTreeWidgetItem* stabilizationItem = findItems(STAB_STRING, Qt::MatchRecursive, 0).value(0);

    QString actionText;
    if (item == exposureItem) {
      actionText = tr("Clear Exposure");
    } else if (item == orientationItem) {
      actionText = tr("Clear Orientation");
    } else if (stabilizationItem && item == stabilizationItem) {
      actionText = tr("Clear Stabilization");
    } else {
      actionText = tr("Clear Keyframes");
    }

    QMenu menu;
    QAction* clear = menu.addAction(actionText);
    QAction* selectedAction = menu.exec(event->globalPos());
    if (selectedAction == clear) {
      resetItem(item);
    }
  }
}

void CurvesTreeWidget::resetItem(QTreeWidgetItem* item) {
  if (item->childCount() == 0) {
    if (item->data(0, Qt::UserRole).canConvert<TimelineItemPayload>()) {
      TimelineItemPayload payload = item->data(0, Qt::UserRole).value<TimelineItemPayload>();
      emit reqResetCurve(payload.curveType, payload.inputId);
    }
  } else {
    for (int i = 0; i < item->childCount(); i++) {
      resetItem(item->child(i));
    }
  }
}

void CurvesTreeWidget::setChildrenSelected(QTreeWidgetItem* item, bool selected) {
  if (item->childCount() == 0) {
    bool parentExpanded = item->parent()->isExpanded() || item->parent() == NULL;
    if (item->parent()->text(0) == STAB_STRING || item->parent()->text(0) == ORIENTATION_STRING) {
      item->setSelected(parentExpanded);
    } else {
      item->setSelected(selected && (userSelectedItems.indexOf(item) >= 0) && parentExpanded);
    }
  } else {
    for (int i = 0; i < item->childCount(); i++) {
      setChildrenSelected(item->child(i), selected);
    }
  }
}
